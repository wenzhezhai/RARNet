"""Read the prepared FSC147-compatible image/exemplar format without training code."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class CountingDataset:
    def __init__(self, root, split):
        self.root = Path(root)
        self.images = self.root / 'images_384_VarV2'
        self.annotations = json.loads((self.root/'annotation_FSC147_384.json').read_text(encoding='utf-8'))
        self.splits = json.loads((self.root/'Train_Test_Val_FSC_147.json').read_text(encoding='utf-8'))
        self.ids = self.splits[split]
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError('Split must contain nonempty, unique image IDs')
        if set(self.splits.get('train', [])) & set(self.splits.get('test', [])):
            raise ValueError('Train/test IDs overlap')
        for name in self.ids:
            target=(self.images/name).resolve()
            if self.images.resolve() not in target.parents or not target.is_file():
                raise ValueError('Missing or unsafe image path: '+name)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        anno = self.annotations[name]
        with Image.open(self.images/name) as image:
            image = image.convert('RGB')
            width, height = image.size
            new_h, new_w = 16*int(height/16), 16*int(width/16)
            if new_h != 384 or new_w < 384:
                raise ValueError('Use the prepared height-384 dataset: '+name)
            image = transforms.ToTensor()(transforms.Resize((new_h,new_w))(image))
        sx = float(new_w)/width
        rects=[];crops=[];scales=[]
        for box in anno['box_examples_coordinates'][:3]:
            x1,y1=int(box[0][0]*sx),int(box[0][1])
            x2,y2=int(box[2][0]*sx),int(box[2][1])
            # Historical inclusive slicing clips an endpoint at the image edge.
            if not (0<=x1<new_w and x1<=x2<=new_w and 0<=y1<new_h and y1<=y2<=new_h):
                raise ValueError('Invalid exemplar box: '+name+' '+str((x1,y1,x2,y2,new_w,new_h)))
            rects.append([y1,x1,y2,x2])
            crops.append(transforms.Resize((64,64))(image[:,y1:y2+1,x1:x2+1]).numpy())
            scales.append([(x2-x1+1)/384,(y2-y1+1)/384])
        if len(crops)!=3:
            raise ValueError('Exactly three exemplar boxes are required: '+name)
        return image,torch.Tensor(np.array(crops)),torch.tensor(scales),rects,name,len(anno['points']) if 'points' in anno else None
