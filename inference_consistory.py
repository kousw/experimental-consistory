import argparse
import itertools
import logging
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

#from accelerate import Accelerator
##from accelerate.logging import get_logger
#from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel, LMSDiscreteScheduler, ControlNetModel, LCMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from omegaconf import OmegaConf

from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from torchvision.transforms import PILToTensor

from typing import Optional, Tuple, Union
import datasets
from datasets import load_dataset
from consistory.models.unet import SDUNet2DConditionModel
from consistory.models.attention import SETransformerBlock

from consistory.models.utils import freeze_params, unfreeze_params, expand_image, image_overlay_heat_map, otsu_thresholding, image_with_otsu
from consistory.pipelines.pipeline_consistory import ConsiStoryPipeline

from extern.dift.dift_sd import SDFeaturizer

def pre_generate_images_for_dift(args):
    height = width = args.resolution

    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
    #lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
    #vae_path = "models/sdxl-vae-fp16"
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.model_name_or_path, subfolder="vae")
    # vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(args.model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_name_or_path, subfolder="text_encoder")
    # unet1 = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet")
    unet = SDUNet2DConditionModel.from_pretrained_vanilla(args.model_name_or_path, subfolder="unet")
    
    pipeline = ConsiStoryPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,        
    )
    
    pipeline.load_lora_weights(lcm_lora_id)
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    
    # pipeline.set_subject_encoder_beta(args.subject_encoder_beta)
    pipeline = pipeline.to(args.device, torch.float16)
    

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        
    batch_size = args.batch_size

    images = []
    prompt = args.prompt
    nagetive_prompt = args.negative_prompt
    if args.reference_image is not None:
        reference_image = Image.open(args.reference_image).convert("RGB")
        reference_image = reference_image.resize((height, width), Image.BILINEAR)
    else:
        reference_image = None
        
    if args.mask_image is not None:
        mask_image = Image.open(args.mask_image)
        mask_image = mask_image.resize((height, width), Image.BILINEAR)
    else:
        mask_image = None
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    
    for _ in range(args.num_samples):
        image = pipeline(
            prompt, height, width, 
            negative_prompt=nagetive_prompt,
            keywords = args.keywords,
            reference_images=reference_image,  
            mask_images=mask_image,
            num_inference_steps=num_inference_steps, 
            generator=generator, 
            guidance_scale=guidance_scale,
            use_vanilla_query = True,
            stop_timestep = 261
        ).images

        images.extend(image)
        
    del pipeline
    

    # Save images
    for i, image in enumerate(images):
        image.save(os.path.join(args.output_dir, "temp_dift", f"image_{i}.png"))
        
    print(len(unet.attention_map.all_attention_maps))
    
    heatmap_images = {}
    attention_words = args.keywords
    
    if args.reference_image is not None:
        batch_size = batch_size + 1

    if guidance_scale > 1.0:
        batch_size = batch_size * 2        
    
    for batch_pos in range(batch_size):
        with torch.no_grad():
            try:
                global_heat_map = unet.attention_map.compute_global_heat_map(tokenizer, prompt, batch_pos, factors=ConsiStoryPipeline.default_attention_map_factors)        
            except Exception as e:
                print(f"Error: {e}")                
                continue
                        
            if global_heat_map is not None:
                img_size = (512, 512)
                caption = ", ".join(attention_words)
                
                heat_map = global_heat_map.compute_word_heat_map(attention_words, mode="sum")
                if heat_map is None : print(f"No heatmaps for '{attention_words}'")
                
                heat_map_img = expand_image(heat_map, img_size[1], img_size[0]) if heat_map is not None else None
                underlay_image = images[batch_pos]
                img : Image.Image = image_overlay_heat_map(underlay_image, heat_map_img, alpha=0.5, caption=caption, image_scale=1.0)
                heat_map_img = heat_map_img.detach().cpu()                    
                img_otsu : Image.Image = image_with_otsu(heat_map_img)

                img.save(os.path.join(args.output_dir, "temp_dift", f"attention_map_{batch_pos}.png"))
                img_otsu.save(os.path.join(args.output_dir, "temp_dift", f"mask_{batch_pos}.png"))
    

def create_dift_map(args):
    
    dift = SDFeaturizer()
        
    guidance_scale = args.guidance_scale
    batch_size = args.batch_size
    if args.reference_image is not None:
        batch_size = batch_size + 1
        
    print(batch_size)
    
    if guidance_scale > 1.0:
        batch_size = batch_size * 2
    
    layers = [0, 1, 2, 3]
    
    feature_layers = {}
    images = []
    
    first_prompt = ", ".join(args.keywords)
    prompts = [first_prompt] + args.prompt    
     
    for layer in layers:

        features = []
    
        for i in range(batch_size):
            img = Image.open(os.path.join(args.output_dir, "temp_dift", f"image_{i}.png")).convert('RGB')
            images.append(img)
            # if args.img_size[0] > 0:
            #     img = img.resize(args.img_size)
            img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2 # Normalize to [-1, 1]

            prompt = prompts[i]
                
            ft = dift.forward(img_tensor,
                            prompt=prompt,
                            t=261,
                            up_ft_index=layer, # choices=[0, 1, 2 ,3]
                        ensemble_size=8)
            
            features.append(ft.squeeze(0))
            
            print(f"ft.shape: {ft.shape}")
            print(f"ft: {ft}")
            
        features = torch.stack(features, 0)
        feature_layers[layer] = features
            
    del dift
    
    # sim_threshold = 0.5
    
    # feature_map_layers = [] # layer
    
    # for layer in layers:
    #     features = feature_layers[layer]
        
    #     feature_map_batch = []  # batch
        
    #     for i, ft in enumerate(features):
    #         # ft = (1, channels, height, width)           
                     
    #         feature_map_h = []  # height
                        
    #         for h in range(ft.shape[2]):
                
    #             feature_map_w = [] # width
                
    #             for w in range(ft.shape[3]):
                    
    #                 max_sim_index = None
    #                 max_sim = 0.0                                       
                    
    #                 p = ft[0, :, h, w]
                    
    #                 for j, ft_j in enumerate(features):
    #                     if j != i:
    #                         for h_j in range(ft_j.shape[2]):
    #                             for w_j in range(ft_j.shape[3]):
    #                                 p_j = ft_j[0, :, h_j, w_j]
                                    
    #                                 # calc cosine similarity
    #                                 sim = torch.dot(p, p_j) / (torch.norm(p) * torch.norm(p_j))
                                    
    #                                 if sim > max_sim and sim_threshold <= sim:
    #                                     max_sim = sim
    #                                     max_sim_index = (j, h_j, w_j)
                                        
    #                 feature_map_w.append(max_sim_index)
                
    #             feature_map_h.append(feature_map_w)
                                        
    #         feature_map_batch.append(feature_map_h)
            
    #     feature_map_layers.append(feature_map_batch)
    
    # cos_thresholdを定義します。
    if args.reference_image is not None:
        cos_threshold = 0.3
    else:
        cos_threshold = 0.5

    def calculate_cosine_similarity_between_batches(features):
        # featuresは、次元が (batch, channels, height, width) のテンソルとします。
        batch, channels, height, width = features.size()
        
        # 特徴マップを (batch, height, width, channels) の形状に変更します。
        features = features.view(batch, channels, -1).transpose(1, 2).view(batch, -1, channels)

        # 各ピクセルベクトルのノルムを計算して正規化します。
        norms = features.norm(dim=2, keepdim=True)
        normalized_features = features / norms
        
        # バッチ間で全てのピクセルベクトルを比較するためにバッチ次元を拡張します。
        normalized_features_expanded = normalized_features.unsqueeze(0)
        normalized_features_tiled = normalized_features.unsqueeze(1)
        
        # すべてのバッチペアの間でコサイン類似度を計算します。
        cosine_similarity = torch.matmul(
            normalized_features_expanded, normalized_features_tiled.transpose(2, 3)
        ).squeeze()
        
        if args.reference_image is not None:
            # リファレンスイメージ以外との類似度を除去します。
            batch_indices = torch.arange(batch).view(batch, 1, 1)
            cosine_similarity[batch_indices, batch_indices, :] = -1
            cosine_similarity[1:batch, 1:batch, :] = -1
        else:
            # 自身との類似度を除去します。
            batch_indices = torch.arange(batch).view(batch, 1, 1)
            cosine_similarity[batch_indices, batch_indices, :] = -1         
                
        print("cosine_similarity shape", cosine_similarity.shape)
        # 類似度が閾値以上の要素のインデックスを取得します。
        max_sim, max_indices = torch.max(cosine_similarity, dim=1)
        print("max_sim, max_indices shape 1", max_sim.shape, max_indices.shape)
        print("max_sim, max_indices",  max_sim, max_indices)
        
        max_sim2, max_indices2 = torch.max(max_sim, dim=1)
        print("max_sim2, max_indices2 shape 2 ", max_sim2.shape, max_indices2.shape)
        print("max_sim2, max_indices2",  max_sim2, max_indices2)
        
        # max_indices : (batch, width * height, width * height)
        # max_indices2 : (batch, width * height)
        
        # final_max_indeces : (batch, height, width, 3) from max_indices and max_indices2        
        final_max_indices = torch.zeros((batch, height, width, 4), dtype=torch.long)

        for b in range(batch):
            for hw in range(height * width):
                                
                h = hw // width
                w = hw % width
                max_hw = max_indices2[b, hw]
                max_h = max_hw // width
                max_w = max_hw % width
                max_b = max_indices[b, hw, max_hw]
                
                cos_sim = max_sim[max_b, max_h, max_w]
                
                if cos_sim < cos_threshold:
                    max_b = -1
                    max_h = -1
                    max_w = -1    
                    
                if args.reference_image is not None and b == 0:
                    max_b = -1
                    max_h = -1
                    max_w = -1           
                
                final_max_indices[b, h, w] = torch.tensor([max_b, max_h, max_w, cos_sim * 1000])

                
        #max_indices_b = max_indices[        
        # max_sim, max_indices = torch.max(max_sim, dim=1)
        # print("max_sim, max_indices shape 3", max_sim.shape, max_indices.shape)
        # print("max_sim, max_indices",  max_sim, max_indices)
        
        # 閾値以上のものだけを残します。
        # max_indices[max_sim < cos_threshold] = -1  # 閾値未満のインデックスは-1に設定
        
        # バッチ内のインデックスを高さと幅のインデックスに変換します。
        # max_indices_batch = max_indices
        # max_indices_h = max_indices2 // width
        # max_indices_w = max_indices2 % width
        # print("max_indices_batch.shape, max_indices_h.shape, max_indices_w.shape", max_indices_batch.shape, max_indices_h.shape, max_indices_w.shape)
        
        # バッチインデックスも含めた結果を返します。
        # max_indices_batch = max_indices #  // (height * width)
        # max_indices_h = max_indices_batch // width
        # max_indices_w = max_indices_batch % width

        print("final_max_indices", final_max_indices)
        print("final_max_indices shape", final_max_indices.shape)
        return final_max_indices


    # 各レイヤーに対して計算を行います。
    feature_map_layers = {}
    for layer, features in feature_layers.items():
        feature_map_layers[layer] = calculate_cosine_similarity_between_batches(features)
        
    def show_feature_map(fml, layer,  imgs, batch_index, height, width):
        fm = fml[layer]
        b, h, w, _ = fm.shape
        
        map_pos = fm[batch_index, height, width]
        print(f"map_pos: {map_pos}")
        
        if False:
            source = imgs[batch_index].resize((w, h))
            target = imgs[map_pos[0]].resize((w, h))
            
            color = (255, 0, 0) 
            source.putpixel((width, height), color) 
            target.putpixel((map_pos[2], map_pos[1]), color) 
            
            source.save(os.path.join(args.output_dir, "temp_dift", f"source_{layer}_{batch_index}_{height}_{width}.png"))
            target.save(os.path.join(args.output_dir, "temp_dift", f"target_{layer}_{batch_index}_{height}_{width}_{map_pos[0]}_{map_pos[1]}_{map_pos[2]}_{map_pos[3]}.png"))
        
        
    # print(fml0.shape) # (batch, 16, 16, 3)
    show_feature_map(feature_map_layers, 0, images, 0, 5, 10)
    show_feature_map(feature_map_layers, 0, images, 1, 5, 10)
    show_feature_map(feature_map_layers, 0, images, 2, 5, 10)
    
    # print(fml1.shape) # (batch, 32, 32, 3)
    show_feature_map(feature_map_layers, 1, images, 0, 10, 20)
    show_feature_map(feature_map_layers, 1, images, 1, 10, 20)
    show_feature_map(feature_map_layers, 1, images, 2, 10, 20)
    
    # print(fml2.shape) # (batch, 64, 64, 3)
    show_feature_map(feature_map_layers, 2, images, 0, 20, 40)
    show_feature_map(feature_map_layers, 2, images, 1, 20, 40)
    show_feature_map(feature_map_layers, 2, images, 2, 20, 40)
    
    # print(fml3.shape) # (batch, 64, 64, 3)    
    show_feature_map(feature_map_layers, 3, images, 0, 20, 20)
    show_feature_map(feature_map_layers, 3, images, 1, 20, 20)
    show_feature_map(feature_map_layers, 3, images, 2, 20, 20)
    
    return feature_map_layers
               
                
          
                    
                    
            
            
            
        
        
    
    

def generate_main(args, feature_map_layers):
    # feature_map_layers = None
    height = width = args.resolution
    
    # sd15
    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
    
    # sdxl
    # lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
    #vae_path = "models/sdxl-vae-fp16"
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.model_name_or_path, subfolder="vae")
    # vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(args.model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_name_or_path, subfolder="text_encoder")
    # unet1 = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet")
    unet = SDUNet2DConditionModel.from_pretrained_vanilla(args.model_name_or_path, subfolder="unet")
    
    pipeline = ConsiStoryPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
    )
    
    pipeline.load_lora_weights(lcm_lora_id)
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    
    # pipeline.set_subject_encoder_beta(args.subject_encoder_beta)
    pipeline = pipeline.to(args.device, torch.float16)
    

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        
    batch_size = args.batch_size

    images = []
    prompt = args.prompt
    nagetive_prompt = args.negative_prompt
    if args.reference_image is not None:
        reference_image = Image.open(args.reference_image).convert("RGB")
        reference_image = reference_image.resize((height, width), Image.BILINEAR)
    else:
        reference_image = None        
        
    if args.mask_image is not None:
        mask_image = Image.open(args.mask_image)
        mask_image = mask_image.resize((height, width), Image.BILINEAR)
    else:
        mask_image = None
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    
    for _ in range(args.num_samples):
        image = pipeline(
            prompt , height, width, 
            negative_prompt=nagetive_prompt,
            keywords = args.keywords,
            reference_images=reference_image,  
            mask_images=mask_image,
            num_inference_steps=num_inference_steps, 
            generator=generator, 
            guidance_scale=guidance_scale,
            feature_map_layers = feature_map_layers,
            feature_injection_timpstep_range = (900, 680),
            use_vanilla_query = True,
        ).images
        
        images.extend(image)
        
    del pipeline

    # Save images
    for i, image in enumerate(images):
        image.save(os.path.join(args.output_dir, f"image_{i}.png"))
        
    print(len(unet.attention_map.all_attention_maps))
    
    heatmap_images = {}
    attention_words = args.keywords
    
    if args.reference_image is not None:
        batch_size = batch_size + 1
    
    if guidance_scale > 1.0:
        batch_size = batch_size * 2        
    
    for batch_pos in range(batch_size):
        with torch.no_grad():
            try:
                global_heat_map = unet.attention_map.compute_global_heat_map(tokenizer, prompt, batch_pos, factors=ConsiStoryPipeline.default_attention_map_factors)             
            except Exception as e:
                print(f"Error: {e}")                
                continue
                        
            if global_heat_map is not None:
                            
                img_size = (512, 512)
                caption = ", ".join(attention_words)
                
                heat_map = global_heat_map.compute_word_heat_map(attention_words, mode="sum")
                if heat_map is None : print(f"No heatmaps for '{attention_words}'")
                
                heat_map_img = expand_image(heat_map, img_size[1], img_size[0]) if heat_map is not None else None
                underlay_image = images[batch_pos]
                img : Image.Image = image_overlay_heat_map(underlay_image, heat_map_img, alpha=0.5, caption=caption, image_scale=1.0)
                heat_map_img = heat_map_img.detach().cpu() 
                img_otsu : Image.Image = image_with_otsu(heat_map_img)

                img.save(os.path.join(args.output_dir,f"attention_map_{batch_pos}.png"))
                img_otsu.save(os.path.join(args.output_dir,f"mask_{batch_pos}.png"))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="models/aom3", help="Model name or path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--keywords", nargs="*", type=str, default=None, help="keywords for the prompt")
    parser.add_argument("--prompt", nargs="*", type=str, default=None, help="Prompt text")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt text")
    parser.add_argument("--reference_image", type=str, default=None, help="Reference image path")
    parser.add_argument("--mask_image", type=str, default=None, help="Mask image path")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Enable xformers memory efficient attention")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--disable_feature_map", action="store_true", help="disable feature map")

    
    args = parser.parse_args()        
        
    if args.seed is None:
        # from int min to max
        args.seed = random.randint(0, sys.maxsize)
    
    # args.keywords = ["cat"]   
    
    # args.prompt = [
    #     "a photo of a cat, eating his food while wearing a hat",  
    #     "a photo of a cat, resting in the wild, wearing his goggles",  
    #     "a photo of a cat, “jumping over a puddle",  
    # ]
    
    if args.keywords is None:
        args.keywords = ["1girl"]
           
    if args.prompt is None:
    
        args.prompt = [
            "1girl, best quality, ultra detailed, sitting on the beach",  
            "1girl, best quality, ultra detailed, resting on the wood",  
            "1girl, best quality, ultra detailed, shoot guns and run the battlefield",  
        ]
        
    print(args.keywords)
    print(args.prompt)
    
    args.batch_size = len(args.prompt)       
    
    #args.reference_image = None
    
    pre_generate_images_for_dift(args)
    torch.cuda.empty_cache()  

    feature_map_layers = create_dift_map(args)
    torch.cuda.empty_cache()  

    generate_main(args, feature_map_layers if not args.disable_feature_map else None)
    torch.cuda.empty_cache()  