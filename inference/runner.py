"""Path-only entry point support; no candidate search, calibration, or training."""
import hashlib,json,math,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from models.RARNet import rarnet_vit_base_patch16
from inference.data import CountingDataset
from inference.protocol import predict

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def run(data_root, checkpoint, output_root, splits=('test',), device=None):
    if not splits or any(split not in ('val', 'test') for split in splits):
        raise ValueError('Only FSC147 val/test splits are supported')
    checkpoint=Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError('Set CHECKPOINT to your trusted RARNet weight file: '+str(checkpoint))
    loaders={split:CountingDataset(data_root,split) for split in splits}
    expected={'val':1286,'test':1190}
    for split,data in loaders.items():
        if len(data)!=expected[split]:
            raise ValueError('Unexpected official split size: '+split)
    if set(loaders[next(iter(loaders))].splits['val']) & set(loaders[next(iter(loaders))].splits['test']):
        raise ValueError('Validation/test overlap')
    device=torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(4);torch.manual_seed(728);np.random.seed(728)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # Only load trusted checkpoint files. Old PyTorch releases use pickle here.
    try: payload=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    except TypeError: payload=torch.load(str(checkpoint),map_location='cpu')
    state=payload['model'] if 'model' in payload else payload
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Checkpoint contains nonfinite tensors')
    model=rarnet_vit_base_patch16().eval()
    model.load_state_dict(state,strict=True);model.to(device)
    del payload,state
    identity=sha256(checkpoint)
    root=Path(__file__).resolve().parents[1]
    hashes={str(p.relative_to(root)):sha256(p) for directory in ['models','inference','util'] for p in (root/directory).rglob('*.py')}
    output=Path(output_root)/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    print('Output:',output,'Device:',device,'Checkpoint SHA256:',identity,flush=True)
    for split,data in loaders.items():
        rows=[];start=time.time();dest=output/split;dest.mkdir()
        with (dest/'predictions.jsonl').open('w',encoding='utf-8') as stream,torch.no_grad():
            for i in range(len(data)):
                image,boxes,scales,rects,name,gt=data[i]
                prediction=predict(model,image[None].to(device),boxes[None].to(device),scales[None].to(device),rects)
                row=dict(image_id=name,gt=gt,**prediction);rows.append(row)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if i%100==0:print(split,i,'/',len(data),flush=True)
        scores={}
        labeled=[r for r in rows if r['gt'] is not None]
        for key in ['count']:
            errors=[r[key]-r['gt'] for r in labeled]
            scores[key]={'mae':sum(map(abs,errors))/len(errors),'rmse':math.sqrt(sum(e*e for e in errors)/len(errors))} if errors else None
        summary=dict(dataset='fsc147',split=split,n=len(rows),labeled_n=len(labeled),metrics=scores,checkpoint_sha256=identity,code_sha256=hashes,protocol='historical',scale=1.0,ecnt_normalization=True,precision='float32',tf32=False,seconds=time.time()-start,split_sha256=sha256(Path(data_root)/'Train_Test_Val_FSC_147.json'))
        (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(split,json.dumps(scores),flush=True)
    return output
