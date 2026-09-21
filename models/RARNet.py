from functools import partial
import torch
import torch.nn as nn
from einops import rearrange,repeat
from timm.models.vision_transformer import PatchEmbed
from models.Block.Blocks import Block
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np

from models.checkpoint_compat import convert_legacy_rarn_state_dict


class ExemplarReliabilityAggregation(nn.Module):
    """Exemplar Reliability Aggregation (ERA).

    ERA preserves exemplar-conditioned response maps, estimates query-dependent
    relative reliability using spatial concentration and cross-exemplar
    consensus, and forms a reliability-weighted aggregate without trainable
    parameters.
    """

    def __init__(self, temperature=0.25, concentration_weight=1.0,
                 consensus_weight=1.0, eps=1e-6):
        super().__init__()
        self.temperature = float(temperature)
        self.concentration_weight = float(concentration_weight)
        self.consensus_weight = float(consensus_weight)
        self.eps = float(eps)

    @staticmethod
    def _spatial_concentration(response_maps, eps=1e-6):
        flat = response_maps.float().flatten(start_dim=-2).clamp_min(eps)
        prob = flat / (flat.sum(dim=-1, keepdim=True) + eps)
        entropy = -(prob * prob.clamp_min(eps).log()).sum(dim=-1)
        entropy = entropy / np.log(float(flat.shape[-1]))
        return (1.0 - entropy).clamp(0.0, 1.0).squeeze(2)

    @staticmethod
    def _extract_exemplar_responses(attention, query_side=24, magnitude=None):
        query_tokens = query_side * query_side
        exemplar_tokens = 4 * 4
        attention = torch.mean(attention, dim=1)
        match = attention[:, query_tokens:, :query_tokens]
        num_exemplars = match.shape[1] // exemplar_tokens
        match = rearrange(
            match, 'b (n t) q -> b n t q', t=exemplar_tokens)
        if magnitude is not None:
            magnitude = magnitude[:, :num_exemplars].to(
                device=match.device, dtype=match.dtype)
            match = match * magnitude.unsqueeze(-1).unsqueeze(-1)
        responses = match.sum(dim=2)
        return rearrange(
            responses, 'b n (w h) -> b n 1 w h',
            w=query_side, h=query_side)

    def _reliability_weights(self, responses):
        eps = self.eps
        num_exemplars = responses.shape[1]
        concentration = self._spatial_concentration(responses, eps=eps)
        flat = responses.flatten(start_dim=2).float().clamp_min(eps)
        flat = flat / (flat.sum(dim=-1, keepdim=True) + eps)
        normalized = F.normalize(flat, p=2, dim=-1, eps=eps)

        if num_exemplars > 1:
            consensus_reference = (
                normalized.sum(dim=1, keepdim=True) - normalized
            ) / float(num_exemplars - 1)
            consensus_reference = F.normalize(
                consensus_reference, p=2, dim=-1, eps=eps)
        else:
            consensus_reference = normalized

        consensus = (
            normalized * consensus_reference
        ).sum(dim=-1).clamp(0.0, 1.0)
        scores = (
            self.concentration_weight * concentration
            + self.consensus_weight * consensus
        )
        temperature = max(self.temperature, eps)
        return torch.softmax(scores / temperature, dim=1)

    def forward(self, attention, query_side=24, magnitude=None,
                return_details=False):
        responses = self._extract_exemplar_responses(
            attention, query_side=query_side, magnitude=magnitude)
        weights = self._reliability_weights(responses).to(
            device=responses.device, dtype=responses.dtype)
        aggregate = (
            responses * weights[:, :, None, None, None]
        ).sum(dim=1)
        if return_details:
            return aggregate, responses, weights
        return aggregate


class OrdinalCountAllocation(nn.Module):
    """Ordinal Count Allocation (OCA).

    OCA conditions multi-resolution decoder features on the ERA aggregate,
    estimates ordered local count mass, and renders the finest-scale mass with
    normalized spatial allocation. The formal RARNet branch is non-recursive.
    """

    ORDINAL_BIASES = (-6.82722770, -11.51291546, -11.51291546)

    def __init__(self):
        super().__init__()
        self.response_projector = nn.Sequential(
            nn.Conv2d(257, 64, kernel_size=1, bias=True),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.ordinal_head = nn.Conv2d(64, 3, kernel_size=1, bias=True)
        self.allocation_head = nn.Conv2d(64, 4, kernel_size=1, bias=True)
        self._initialize_heads()

    def _initialize_heads(self):
        nn.init.normal_(self.ordinal_head.weight, std=0.01)
        with torch.no_grad():
            self.ordinal_head.bias.copy_(torch.tensor(
                self.ORDINAL_BIASES, dtype=self.ordinal_head.bias.dtype))
        nn.init.zeros_(self.allocation_head.weight)
        nn.init.zeros_(self.allocation_head.bias)

    @staticmethod
    def ordinal_probabilities(logits):
        conditional = torch.sigmoid(logits.float())
        p1 = conditional[:, 0:1]
        p2 = p1 * conditional[:, 1:2]
        p3 = p2 * conditional[:, 2:3]
        return torch.cat((p1, p2, p3), dim=1)

    def condition(self, feature, aggregate_response):
        response = F.interpolate(
            aggregate_response, size=feature.shape[-2:],
            mode='bilinear', align_corners=False)
        return self.response_projector(torch.cat((feature, response), dim=1))

    def predict_local_mass(self, conditioned_feature):
        logits = self.ordinal_head(conditioned_feature)
        probabilities = self.ordinal_probabilities(logits)
        local_mass = probabilities[:, 0:1] + probabilities[:, 1:2]
        overflow = probabilities[:, 2:3]
        return local_mass, overflow, logits

    def render_density(self, local_mass, finest_conditioned_feature):
        allocation_weights = torch.softmax(
            self.allocation_head(finest_conditioned_feature).float(), dim=1)
        density = 60.0 * F.pixel_shuffle(
            local_mass * allocation_weights, upscale_factor=2)
        return density, allocation_weights


class RARNet(nn.Module):
    """Reliability-Aware Readout Network (RARNet)."""

    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 era_temperature=0.25, era_concentration_weight=1.0,
                 era_consensus_weight=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        ex_size = 64

        ## Encoder specifics
        self.scale_embeds = nn.Linear(2,embed_dim,bias = True)
        self.patch_embed_exemplar = PatchEmbed(ex_size, patch_size, in_chans+1, embed_dim)
        num_patches_exemplar = self.patch_embed_exemplar.num_patches
        self.pos_embed_exemplar = nn.Parameter(torch.zeros(1, num_patches_exemplar, embed_dim), requires_grad=False)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.norm = norm_layer(embed_dim)

        self.blocks = nn.ModuleList([
        Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        for i in range(depth)])

        # Retained parameter slots allow strict loading of existing checkpoints.
        self.v_y = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        self.density_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_exemplar = nn.Parameter(torch.zeros(1, num_patches_exemplar, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_norm = norm_layer(decoder_embed_dim)
        ### decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        ### decoder blocks
        ## Decoder specifics
        ## Regressor
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(513, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            # Retained checkpoint slot; inference uses the OCA density output.
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )
        ## Regressor

        self.initialize_weights()
        self.era = ExemplarReliabilityAggregation(
            temperature=era_temperature,
            concentration_weight=era_concentration_weight,
            consensus_weight=era_consensus_weight,
        )
        self.oca = OrdinalCountAllocation()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_embde_exemplar = get_2d_sincos_pos_embed(self.pos_embed_exemplar.shape[-1], int(self.patch_embed_exemplar.num_patches**.5), cls_token=False)
        self.pos_embed_exemplar.copy_(torch.from_numpy(pos_embde_exemplar).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        decoder_pos_embed_exemplar = get_2d_sincos_pos_embed(self.decoder_pos_embed_exemplar.shape[-1], int(self.patch_embed_exemplar.num_patches**.5), cls_token=False)
        self.decoder_pos_embed_exemplar.data.copy_(torch.from_numpy(decoder_pos_embed_exemplar).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w1 = self.patch_embed_exemplar.proj.weight.data
        torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def scale_embedding(self, exemplars, scale_infos):
        bs, n, c, h, w = exemplars.shape
        scales_batch = []
        for i in range(bs):
            scales = []
            for j in range(n):
                w_scale = torch.linspace(0,scale_infos[i,j,0],w)
                w_scale = repeat(w_scale,'w->h w',h=h).unsqueeze(0)
                h_scale = torch.linspace(0,scale_infos[i,j,1],h)
                h_scale = repeat(h_scale,'h->h w',w=w).unsqueeze(0)
                scale = w_scale + h_scale
                scales.append(scale)
            scales = torch.stack(scales)
            scales_batch.append(scales)
        scales_batch = torch.stack(scales_batch)

        scales_batch = scales_batch.to(exemplars.device)
        exemplars = torch.cat((exemplars,scales_batch),dim=2)

        return exemplars

    def forward_encoder(self, x, y, scales=None):
        y_embed = []
        y = rearrange(y,'b n c w h->n b c w h')
        for box in y:
            box = self.patch_embed_exemplar(box)
            box = box + self.pos_embed_exemplar
            y_embed.append(box)
        y_embed = torch.stack(y_embed, dim=0)
        box_num,_,n,d = y_embed.shape
        y = rearrange(y_embed, 'box_num batch n d->batch (box_num  n) d')
        x = self.patch_embed(x)
        x = x + self.pos_embed
        _, l, d = x.shape
        attns = []
        x_y = torch.cat((x,y),axis=1)
        for i, blk in enumerate(self.blocks):
            x_y, attn = blk(x_y)
            attns.append(attn)
        x_y = self.norm(x_y)
        x = x_y[:,:l,:]
        for i in range(box_num):
            y[:,i*n:(i+1)*n,:] = x_y[:,l+i*n:l+(i+1)*n,:]
        y = rearrange(y,'batch  (box_num  n) d->box_num batch n d',box_num = box_num,n=n)
        return x, y

    def forward_decoder(self,x,y,scales=None):
        x = self.decoder_embed(x)
        # add pos embed
        x = x + self.decoder_pos_embed
        b,l_x,d = x.shape
        y_embeds = []
        num, batch, l, dim = y.shape
        for i in range(num):
            y_embed = self.decoder_embed(y[i])
            y_embed = y_embed + self.decoder_pos_embed_exemplar
            y_embeds.append(y_embed)
        y_embeds = torch.stack(y_embeds)
        num, batch, l, dim = y_embeds.shape
        y_embeds = rearrange(y_embeds,'n b l d -> b (n l) d')
        x = torch.cat((x,y_embeds),axis=1)
        attns = []
        xs = []
        ys = []
        for i, blk in enumerate(self.decoder_blocks):
            x, attn = blk(x)
            if i == 2:
                x = self.decoder_norm(x)
            attns.append(attn)
            xs.append(x[:,:l_x,:])
            ys.append(x[:,l_x:,:])
        return xs,ys,attns


    def build_matching_feature(self, xs, ys, attn, scales=None):
        x = xs[-1]
        _, query_length, _ = x.shape
        r2 = (scales[:, :, 0] + scales[:, :, 1]) ** 2
        magnitude = 16 / (r2 * 384)
        density_feature = rearrange(
            x, 'b (w h) d -> b d w h', w=int(np.sqrt(query_length)))
        aggregate_response = self.era(
            attn[-1], query_side=int(np.sqrt(query_length)),
            magnitude=magnitude)
        return torch.cat(
            (density_feature.contiguous(), aggregate_response.contiguous()),
            dim=1)

    def decoder_feature(self, feature):
        """Compute the finest decoder feature used by the density readout."""
        feature0 = self.decode_head0(feature)
        feature1 = self.decode_head1(F.interpolate(
            feature0, size=feature0.shape[-1] * 2,
            mode='bilinear', align_corners=False))
        feature2 = self.decode_head2(F.interpolate(
            feature1, size=feature1.shape[-1] * 2,
            mode='bilinear', align_corners=False))
        feature3_input = F.interpolate(
            feature2, size=feature2.shape[-1] * 2,
            mode='bilinear', align_corners=False)
        feature3 = self.decode_head3[0](feature3_input)
        feature3 = self.decode_head3[1](feature3)
        feature3 = self.decode_head3[2](feature3)
        return feature3

    def oca_readout(self, density_feature):
        feature = self.decoder_feature(density_feature)
        conditioned = self.oca.condition(feature, density_feature[:, -1:, :, :])
        local_mass, _, _ = self.oca.predict_local_mass(conditioned)
        density, _ = self.oca.render_density(local_mass, conditioned)
        return density.squeeze(1)

    @torch.no_grad()
    def forward(self, samples):
        imgs, boxes, scales = samples
        boxes = self.scale_embedding(boxes, scales)
        latent, exemplar_latent = self.forward_encoder(imgs, boxes, scales=scales)
        xs, ys, attns = self.forward_decoder(latent, exemplar_latent)
        density_feature = self.build_matching_feature(xs, ys, attns, scales)
        return self.oca_readout(density_feature)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = convert_legacy_rarn_state_dict(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:  # PyTorch versions before ``assign`` was added.
            return super().load_state_dict(state_dict, strict=strict)


def rarnet_vit_base_patch16():
    """Complete RARNet with ERA and OCA permanently enabled."""
    return RARNet(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
