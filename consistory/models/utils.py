import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib import cm
from fonts.ttf import Roboto

def freeze_params(params):
    for param in params:
        param.requires_grad_(False)

def unfreeze_params(params):
    for param in params:
        param.requires_grad_(True)
        
def expand_image(im: torch.Tensor, h = 512, w = 512,  absolute: bool = False, threshold: float = None) -> torch.Tensor:

    im = im.unsqueeze(0).unsqueeze(0)
    im = F.interpolate(im.float().detach(), size=(h, w), mode='bicubic')

    if not absolute:
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)

    if threshold:
        im = (im > threshold).float()

    # im = im.cpu().detach()

    return im.squeeze()

def image_overlay_heat_map(img, heat_map, word=None, out_file=None, crop=None, alpha=0.5, caption=None, image_scale=1.0):
    # type: (Image.Image | np.ndarray, torch.Tensor, str, Path, int, float, str, float) -> Image.Image
    assert(img is not None)

    if heat_map is not None:
        shape : torch.Size = heat_map.shape
        # heat_map = heat_map.unsqueeze(-1).expand(shape[0], shape[1], 3).clone()
        heat_map = _convert_heat_map_colors(heat_map)
        heat_map = heat_map.to('cpu').detach().numpy().copy().astype(np.uint8)
        heat_map_img = Image.fromarray(heat_map)

        img = Image.blend(img, heat_map_img, alpha)
    else:
        img = img.copy()

    if caption:
        img = _write_on_image(img, caption)

    if image_scale != 1.0:
        x, y = img.size
        size = (int(x * image_scale), int(y * image_scale))
        img = img.resize(size, Image.BICUBIC)

    return img

def otsu_thresholding(image):
    # 画像のヒストグラムを計算
    hist, bin_edges = np.histogram(image, bins=256, range=(0, 255))

    # 各閾値でのクラス内分散とクラス間分散を計算
    pixel_sum = np.sum(hist)
    weight1 = np.cumsum(hist)
    weight2 = pixel_sum - weight1

    # ゼロ除算を避ける
    mean1 = np.cumsum(hist * bin_edges[:-1]) / (weight1 + (weight1 == 0))
    mean2 = (np.cumsum(hist[::-1] * bin_edges[1:][::-1]) / (weight2[::-1] + (weight2[::-1] == 0)))[::-1]

    # クラス間分散を最大にする閾値を求める
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    threshold = np.argmax(inter_class_variance)

    # 二値化処理
    binary_image = np.where(image <= threshold, 0, 255)
    return binary_image.astype(np.uint8)

def otsu_thresholding_torch(image):
    # 画像のヒストグラムを計算
    hist = torch.histc(image.float(), bins=256, min=0, max=255).to(image.device)

    # 各閾値でのクラス内分散とクラス間分散を計算
    pixel_sum = torch.sum(hist)
    weight1 = torch.cumsum(hist, 0)
    weight2 = pixel_sum - weight1

    bin_edges = torch.arange(256).float().to(image.device)

    # ゼロ除算を避ける
    mean1 = torch.cumsum(hist * bin_edges, 0) / (weight1 + (weight1 == 0))
    mean2 = (torch.cumsum(hist.flip(0) * bin_edges.flip(0), 0) / (weight2.flip(0) + (weight2.flip(0) == 0))).flip(0)


    # クラス間分散を最大にする閾値を求める
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    threshold = torch.argmax(inter_class_variance)

    # 二値化処理
    binary_image = torch.where(image <= threshold, 0, 255)
    return binary_image.type(torch.uint8)

def image_with_otsu(tensor):
    # Tensorをnumpy配列に変換し、範囲を0-255に変換
    image = np.array(tensor * 255, dtype=np.uint8)

    # 大津の二値化を適用
    binary_image = otsu_thresholding(image)

    # 二値化された画像をPIL Imageオブジェクトに変換
    return Image.fromarray(binary_image)

def mask_with_otsu(tensor):
    # Tensorをnumpy配列に変換し、範囲を0-255に変換
    image = np.array(tensor * 255, dtype=np.uint8)

    # 大津の二値化を適用
    binary_image = otsu_thresholding(image)

    # 二値化された画像を0~1のTensorに変換
    return torch.tensor(binary_image / 255, dtype=tensor.dtype)


def mask_with_otsu_pytorch(tensor : torch.Tensor):
    # Tensorをnumpy配列に変換し、範囲を0-255に変換
    image = (tensor * 255).to(torch.uint8)

    # 大津の二値化を適用
    binary_image = otsu_thresholding_torch(image)

    # 二値化された画像を0~1のTensorに変換
    return (binary_image / 255).to(tensor.dtype)

def _write_on_image(img, caption, font_size = 32):
    ix,iy = img.size
    draw = ImageDraw.Draw(img)
    margin=2
    fontsize=font_size
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(Roboto, fontsize)
    text_height=iy-60
    tx = draw.textbbox((0,0),caption,font)
    draw.text((int((ix-tx[2])/2),text_height+margin),caption,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2),text_height-margin),caption,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2+margin),text_height),caption,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2-margin),text_height),caption,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2),text_height), caption,(255,255,255),font=font)
    return img



def _convert_heat_map_colors(heat_map : torch.Tensor):
    def get_color(value):
        return np.array(cm.turbo(value / 255)[0:3])

    color_map = np.array([ get_color(i) * 255 for i in range(256) ])
    color_map = torch.tensor(color_map, device=heat_map.device, dtype=heat_map.dtype)

    heat_map = (heat_map * 255).long()

    return color_map[heat_map]