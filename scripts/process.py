import os
import glob
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from rembg import remove
from transparent_background import Remover
from tqdm import tqdm

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--rgb', default=True, type=bool, help="output rgb image")
    parser.add_argument('--depth', default=True, type=bool, help="output depth image")
    parser.add_argument('--mask', default=True, type=bool, help="output mask image")
    parser.add_argument('--face', default=False, type=bool, help="process faces")
    parser.add_argument('--limit', default=100, type=int, help="output resolution")
    parser.add_argument('--remover', default='inspy', type=str, help="remover type")
    parser.add_argument('--threshold', default=None, type=float, help="remover threshold")
    parser.add_argument('--size', default=512, type=int, help="output resolution")

    opt = parser.parse_args()
    
    device = "cuda"

    # Load remover model
    if opt.remover == 'inspy':
        remover = Remover(device=device) 
    elif opt.remover == 'rembg':
        remover = remove
    else:
        raise NotImplementedError()
    # remover = Remover(mode='fast', jit=True, device='cuda:0', ckpt='~/latest.pth') # custom setting
    # remover = Remover(mode='base-nightly') # nightly release checkpoint
    
    # load midas model
    model_type = "DPT_Large" 
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        midas_transform = midas_transforms.dpt_transform
    else:
        midas_transform = midas_transforms.small_transform

    if os.path.isdir(opt.path):
        print(f'[INFO] processing directory {opt.path}...')
        if opt.face:
            files = glob.glob(f'{opt.path}/*[0-9]_face.png')
        else:
            files = glob.glob(f'{opt.path}/*[0-9].png')
        out_dir = opt.path
    else: # isfile
        files = [opt.path]
        out_dir = os.path.dirname(opt.path)
        
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
       
    limit = opt.limit
    if limit < len(files):
        limit = len(files)
        
    # limitまで
    files = files[:opt.limit]
    with torch.no_grad():
        for file in tqdm(files):

            out_base = os.path.basename(file).split('.')[0]
            out_rgb = os.path.join(out_dir, out_base + '_rgb.png')
            out_depth = os.path.join(out_dir, out_base + '_depth.png')
            out_mask = os.path.join(out_dir, out_base + '_mask.png')

            # load image
            print(f'[INFO] loading image {file}...')
            img = Image.open(file).convert('RGB')
            
            # carve background
            print(f'[INFO] background removal...')
            if opt.remover == 'inspy':            
                carved_image = remover.process(img, type='rgba', threshold=opt.threshold)
            elif opt.remover == 'rembg':
                carved_image = remover(img)
            else:
                raise NotImplementedError()
            
            
            if opt.depth:
                img = pil2cv(carved_image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                input_batch = midas_transform(img).to(device)            
                prediction = midas(input_batch)

                size = (carved_image.size[1], carved_image.size[0])

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=size,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                depth_image = prediction.cpu().numpy()
                        
            # use alpha channel in carved_image for mask image, use rgb in carved_image for rgb image
            if opt.rgb or opt.mask:
                carved_image = np.array(carved_image)
                mask = carved_image[:,:,3]
                rgb = carved_image[:,:,:3]
                
                # apply mask blending as white background
                # アルファチャンネルを取得し、正規化する
                mask_normalized = mask / 255.0

                # 白い背景画像を作成
                white_background = np.ones_like(rgb, dtype=np.uint8) * 255

                # RGBチャンネルと白い背景をアルファチャンネルでブレンド
                rgb = (rgb * mask_normalized[:, :, None]) + (white_background * (1 - mask_normalized[:, :, None]))
                
                # save mask
                print(f'[INFO] saving mask...')
                mask = Image.fromarray(mask)
                rgb = Image.fromarray(rgb.astype(np.uint8))
            
            # save rgb and depth
            print(f'[INFO] saving rgb, mask, and depth...')
            if opt.rgb:
                rgb.save(out_rgb)
            if opt.mask:
                mask.save(out_mask)
            if opt.depth:           
                print("depth min/max", depth_image.min(), depth_image.max())                        
                depth_image = Image.fromarray((depth_image / depth_image.max() * 255).astype(np.uint8))
                depth_image.save(out_depth)
