import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import skimage.io as io
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import cv2
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import pycocotools._mask as _mask
from pycocotools.coco import COCO
import pycocotools._mask as _mask
import json
import clip

from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

import os
import random
import numpy as np
from PIL import Image
from pycocotools.coco import COCO




class ROSD(data.Dataset):
    def __init__(self,
                 split='test', 
                 image_transforms=None,
                 target_transforms=None,
                 eval_mode=False):
        self.eval_mode = eval_mode
        self.image_transforms = _transform(args.img_size)
        self.target_transform = target_transforms
        self.img_size = 480
        self.imgfile = ''
        self.annfile=''
        self.coco=COCO(self.annfile)
    
    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, index):
        eval_data_json = self.data_json[index]
        with open(eval_data_json, 'r') as f:
            infos = json.load(f)
            image_id = infos['image_id']
            img = self.coco.loadImgs(image_id)[0]
            I = Image.open(os.path.join(self.imgfile, img['file_name'])).convert("RGB").resize((self.img_size, self.img_size))
            I = self.image_transforms(I)
            expression = infos['expression']
            expression_encode = clip.tokenize(expression)
            expression_encode = expression_encode.squeeze(0)

            target_bboxes = []
            selected_masks = infos['target_annotation']
            for key in selected_masks:
                if 'counts' in selected_masks[key]:
                    mask_info = {
                        'image_id': image_id,
                        'segmentation':{
                        'counts': list(selected_masks[key]['counts']),
                        'size': selected_masks[key]['size']
                    }}
                else:
                    mask_info = {
                        'image_id': image_id, 
                        'segmentation': list(selected_masks[key])
                    }
                mask = self.coco.annToMask(mask_info)
                mask = cv2.resize(mask, (self.img_size, self.img_size))
                mask = (mask * 255).astype(np.uint8)

                mask_binary = (mask > 0).astype(np.uint8)
                non_zero_indices = np.argwhere(mask_binary)
                y_min, x_min = non_zero_indices.min(axis=0)
                y_max, x_max = non_zero_indices.max(axis=0)
                c_x = float((x_min + x_max) / 2)
                c_y = float((y_min + y_max) / 2)
                w = float(x_max - x_min)
                h = float(y_max - y_min)
                bbox= (x_min/self.img_size, y_min/self.img_size, w/self.img_size, h/self.img_size)
                target_bboxes.append(bbox)
            num_inst = len(target_bboxes)
            max_bbox = 20
            if num_inst < max_bbox:
                for i in range(0, max_bbox-num_inst):
                    target_bboxes.append([0.0, 0.0, 0.0, 0.0])     
        return os.path.join(self.imgfile, img['file_name']),I, expression_encode, torch.tensor(target_bboxes), {'expression': expression,'image_id': image_id, 'index':index, 'num_inst':num_inst}        
