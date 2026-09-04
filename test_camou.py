import os
import torch
import cv2
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from diffusers import StableDiffusionXLInpaintPipeline
from style_adapter import StyleAdapterXLInpaint as StyleAdapter
import glob
import argparse
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from controlnet_aux import ContentShuffleDetector
from scipy.ndimage import binary_dilation, binary_erosion


def ripple_pad(image, mask, pad_width=16):
    image_np = np.array(image)
    mask_np = np.array(mask)
    object_mask = mask_np > 127
    object_mask = binary_erosion(object_mask, iterations=3)
    dilated_mask = binary_dilation(object_mask, iterations=pad_width)
    boundary_mask = dilated_mask ^ object_mask
    padded_image = np.array(image).copy()
    from scipy.ndimage import distance_transform_edt
    distance, indices = distance_transform_edt(~object_mask, return_indices=True)
    for c in range(3):
        channel = image_np[:, :, c]
        nearest_values = channel[indices[0], indices[1]]
        padded_image[:, :, c][boundary_mask] = nearest_values[boundary_mask]
    return Image.fromarray(padded_image)


def copy_random_patches(image, mask, patch_size=32):
    image_np = np.array(image)
    mask_np = np.array(mask)
    image_np = image_np * (mask_np[:, :, np.newaxis] // 255)
    masked_coords = np.argwhere(mask_np > 127)
    valid_patches = [
        (y, x) for y, x in masked_coords
        if y + patch_size <= image_np.shape[0]
        and x + patch_size <= image_np.shape[1]
        and np.all(mask_np[y:y+patch_size, x:x+patch_size] > 127)
    ]
    if not valid_patches:
        return None
    new_image_np = np.zeros_like(image_np)
    for y in range(0, image_np.shape[0], patch_size):
        for x in range(0, image_np.shape[1], patch_size):
            patch_y, patch_x = random.choice(valid_patches)
            patch = image_np[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size]
            end_y = min(y + patch_size, image_np.shape[0])
            end_x = min(x + patch_size, image_np.shape[1])
            patch_height = end_y - y
            patch_width = end_x - x
            new_image_np[y:end_y, x:end_x] = patch[:patch_height, :patch_width]
    return Image.fromarray(new_image_np)


def resize_img(
    input_image,
    max_side=1024,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)
    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y: offset_y + h_resize_new, offset_x: offset_x + w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def get_file_list(input_path):
    """兼容输入：单文件路径 / 文件夹路径，返回图片路径列表"""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        exts = ("*.jpg", "*.jpeg", "*.png")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(input_path, ext)))
        return sorted(files)
    else:
        raise FileNotFoundError(f"{input_path} is not file or directory")


def find_corresponding_mask(img_path, mask_suffix=".png"):
    """图片 a.jpg → mask a.png"""
    base, _ = os.path.splitext(img_path)
    mask_path = base + mask_suffix
    if os.path.exists(mask_path):
        return mask_path
    return None


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load blip2
    processor = AutoProcessor.from_pretrained(args.blip2_model_path)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        args.blip2_model_path, torch_dtype=torch.float16
    )
    blip_model.to(device)

    # sdxl inpaint pipe
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.inpaint_model_path, torch_dtype=torch.float16
    ).to(device)

    ip_model = StyleAdapter(pipe, args.ip_ckpt, device)
    image_processor = ContentShuffleDetector()

    add_prompt = "high quality, cinematic photo, cinemascope, 35mm, film grain, highly detailed"
    # n_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, painting"

    os.makedirs(args.output_dir, exist_ok=True)
    img_list = get_file_list(args.input_path)
    print(f"Total {len(img_list)} images to process")

    for img_path in img_list:
        mask_path = find_corresponding_mask(img_path)
        if mask_path is None:
            print(f"Skip {img_path}, cannot find corresponding mask file")
            continue

        basename = os.path.basename(img_path)
        print(f"Processing: {basename}")

        with torch.no_grad():
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            image = resize_img(image)
            w, h = image.size
            mask = mask.resize((w, h), Image.NEAREST)
            mask = mask.point(lambda p: 255 if p >= 128 else 0)

            # BLIP‑2 get object category
            blip_prompt = "Question: What is the object in the image? Answer:"
            mask_inv = ImageOps.invert(mask)
            mask_object = Image.composite(image, Image.new(mode="RGB", size=(w, h), color=(255, 255, 255)), mask_inv)
            inputs = processor(images=mask_object, text=blip_prompt, return_tensors="pt").to(device, dtype=torch.float16)
            generated_ids = blip_model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            cat = generated_text.replace("Question: What is the object in the image? Answer:", "")\
                .replace("It is ", "").replace("it is ", "").replace("It's ", "").replace("The object is ", "")
            print(f"Detected category: {cat}")

            if args.shuffle:
                init_image = copy_random_patches(image, mask_inv)
                if init_image is None:
                    init_image = copy_random_patches(image, mask_inv, 16)
                if init_image is None:
                    init_image = copy_random_patches(image, mask_inv, 8)
                if init_image is None:
                    init_image = copy_random_patches(image, mask_inv, 4)

                shuffle_image = image_processor(init_image)
                shuffle_image = shuffle_image.resize((w, h))
                pad_image = ripple_pad(image, mask_inv, pad_width=32)
                dilated_mask = mask_inv.filter(ImageFilter.MaxFilter(size=29))
                shuffle_image2 = Image.composite(pad_image, shuffle_image, dilated_mask)
            else:
                shuffle_image2 = image

            image_prompt = cat + ", " + add_prompt
            generated_image = ip_model.generate(
                pil_image=shuffle_image2,
                mask_image=mask,
                prompt=image_prompt,
                num_samples=1,
                scale=0,
                num_inference_steps=args.infer_steps,
                seed=args.seed,
                width=w,
                height=h
            )
            save_path = os.path.join(args.output_dir, basename)
            generated_image[0].save(save_path)
            print(f"Saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single / folder test for camouflage generation")
    parser.add_argument("--input_path", type=str, required=True,
                        help="single image file OR image folder; mask should be same name .png")
    parser.add_argument("--output_dir", type=str, required=True, help="output save directory")
    parser.add_argument("--blip2_model_path", type=str, default="Salesforce/blip2-opt-6.7b")
    parser.add_argument("--inpaint_model_path", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--ip_ckpt", type=str, required=True, help="ip_adapter.bin checkpoint path")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--infer_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
