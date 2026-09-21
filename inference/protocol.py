"""Fixed RARNet inference protocols. No target labels or training imports."""
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from inference.windows import window_starts, merge_windows

def protocol_count(density, rects):
    count = float((density / 60).sum())
    ecnt = sum(float((density[r[0]:r[2]+1, r[1]:r[3]+1] / 60).sum()) for r in rects[:3]) / 3
    return count / ecnt if ecnt > 1.8 else count


def historical_crops(image, divisions):
    height, width = image.shape[-2:]
    positions = ((0,0),(1,0),(0,1),(1,1),(2,0),(2,1),(0,2),(1,2),(2,2)) if divisions == 3 else (
        (0,0),(1,0),(0,1),(1,1))
    for row, col in positions:
        crop_h, crop_w = int(height/divisions), int(width/divisions)
        crop = TF.crop(image[0], int(height*row/divisions), int(width*col/divisions), crop_h, crop_w)
        yield transforms.Resize((height, width))(crop).unsqueeze(0), width/crop_w, height/crop_h


@torch.no_grad()
def original_map(model, image, boxes, scales):
    if image.shape[-2] != 384:
        raise ValueError('Prepared images must have height 384.')
    starts = window_starts(image.shape[-1])
    patches = [model([image[..., x:x+384], boxes, scales]).squeeze(0).float()
               for x in starts]
    return merge_windows(patches, starts, image.shape[-1])


@torch.no_grad()
def predict(model, image, boxes, scales, rects):
    """Historical FSC147 protocol, without scale fusion."""
    count = protocol_count(original_map(model, image, boxes, scales), rects)
    small = any(r[2]-r[0] < 10 and r[3]-r[1] < 10 for r in rects[:3])
    if small:
        tiled = sum(float((original_map(model, crop, boxes, scales)/60).sum())
                    for crop, _, _ in historical_crops(image, 3))
        count = count if tiled > 9*count else tiled
    if not np.isfinite(count):
        raise ValueError('Nonfinite prediction')
    return {'count': count, 'small_exemplar': small}
