import os
import glob

from PIL import Image
import numpy as np
import torch
import tqdm
import argparse

from extern.TorchDeepDanbooru import deep_danbooru_model 
from extern.TorchDeepDanbooru.image_utils import transform_and_pad_image 

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def expand2square(pil_img, background_color = (255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--face', default=False, type=bool, help="process faces")
    parser.add_argument('--top', default=20, type=int, help="output top tags")
    parser.add_argument('--limit', default=100, type=int, help="limit number of images to process")
    opt = parser.parse_args()
    
    device = "cuda"
    
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

    model = deep_danbooru_model.DeepDanbooruModel()
    model.load_state_dict(torch.load('./models/deepdanbooru/model-resnet_custom_v3.pt'))

    model.eval()
    model.to(device)
    # model.half()
    # model.cuda()
    
    for file in tqdm.tqdm(files):
        # without extension
        basename = os.path.basename(file)
        file_without_ext = basename.split('.')[0]
            
        image = Image.open(file).convert("RGB")
        
        width = 512 
        height = 512
        
        image = expand2square(image)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        # image.save('test.png')
        image = np.array(image)
        image = image / 255.0
        image_shape = image.shape
        image = image.reshape((1, image_shape[0], image_shape[1], image_shape[2]))
        
        
        with torch.no_grad():
            x = torch.from_numpy(image).to(device).to(torch.float32)
            # first run
            y = model(x)[0].detach().cpu().numpy()

            # measure performance
            # for n in tqdm.tqdm(range(10)):
            #     model(x)

        # sort tags by probability 
        dict = list(zip(model.tags, y))
        dict.sort(key=lambda x: x[1], reverse=True)
        # print(dict[:opt.top])
              
        # get top tags and join them with space
        # exclude tag that starts with "rating:"
        tags = ','.join([p[0] for p in dict[:opt.top] if not p[0].startswith('rating:')] )
        
        # save as json file
        with open(f'{out_dir}/{os.path.basename(file_without_ext)}_tags.json', 'w') as f:
           f.write(f'{{"general": "{tags}"}}')
