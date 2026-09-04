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


def ripple_effect(image, mask, pad_width=8, num_ripples=3):
    image_np = np.array(image)
    mask_np = np.array(mask) > 127
    output = image_np.copy()

    current_mask = mask_np.copy()
    for i in range(num_ripples):
        next_mask = binary_dilation(current_mask, iterations=pad_width)
        ripple_band = next_mask ^ current_mask

        from scipy.ndimage import distance_transform_edt
        distance, indices = distance_transform_edt(~current_mask, return_indices=True)

        for c in range(3):
            channel = output[:, :, c]
            nearest_values = channel[indices[0], indices[1]]
            channel[ripple_band] = nearest_values[ripple_band]
            output[:, :, c] = channel

        current_mask = next_mask
    return Image.fromarray(output)


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

    new_image = Image.fromarray(new_image_np)
    return new_image


def randomize_seed_fn(seed: int, randomize_seed: bool, max_seed: int) -> int:
    if randomize_seed:
        seed = random.randint(0, max_seed)
    return seed


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


def pil_to_cv2(image_pil):
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2


def crop_mask(image, image1, mask, new_size=512):
    image = np.array(image)
    image1 = np.array(image1)
    mask = np.array(mask)

    height, width = mask.shape
    scaling_factor = new_size / min(height, width)
    mask = cv2.resize(mask, (int(width * scaling_factor), int(height * scaling_factor)), interpolation=cv2.INTER_AREA)
    mask[mask > 127] = 255
    mask[mask < 128] = 0
    image = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)), interpolation=cv2.INTER_AREA)
    image1 = cv2.resize(image1, (int(width * scaling_factor), int(height * scaling_factor)), interpolation=cv2.INTER_AREA)

    x, y, w, h = cv2.boundingRect(mask)
    center_x, center_y = x + w // 2, y + h // 2
    start_x = max(center_x - new_size // 2, 0)
    start_y = max(center_y - new_size // 2, 0)
    end_x = min(start_x + new_size, mask.shape[1])
    end_y = min(start_y + new_size, mask.shape[0])
    start_x = max(end_x - new_size, 0)
    start_y = max(end_y - new_size, 0)

    cropped_mask = mask[start_y:end_y, start_x:end_x]
    cropped_img = image[start_y:end_y, start_x:end_x]
    cropped_img1 = image1[start_y:end_y, start_x:end_x]
    return Image.fromarray(cropped_img), Image.fromarray(cropped_img1), Image.fromarray(cropped_mask)


def main(args):
    MAX_SEED = np.iinfo(np.int32).max
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

    # load blip2
    processor = AutoProcessor.from_pretrained(args.blip2_model_path)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        args.blip2_model_path, torch_dtype=torch.float16
    )
    blip_model.to(device)

    # load sdxl inpaint pipeline
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.inpaint_model_path,
        torch_dtype=torch.float16,
    ).to(device)

    ip_model = StyleAdapter(pipe, args.ip_ckpt, device)
    image_processor = ContentShuffleDetector()

    add_prompt = "high quality, cinematic photo, cinemascope, 35mm, film grain, highly detailed"
    n_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, painting"

    for data_type in ["COD", "SOD", "SEG"]:
        masks = sorted(glob.glob(os.path.join(args.dataset_root, 'validation/masks', data_type + "_*")))
        images = sorted(glob.glob(os.path.join(args.dataset_root, 'validation/images', data_type + "_*")))

        out_dir = os.path.join(args.dst_root, data_type)
        os.makedirs(out_dir, exist_ok=True)

        # resume
        names = os.listdir(out_dir)
        m_finish = [os.path.join(args.dataset_root, 'validation/masks', i.replace('.jpg', '.png')) for i in names]
        i_finish = [os.path.join(args.dataset_root, 'validation/images', i) for i in names]
        masks = [i for i in masks if i not in m_finish]
        images = [i for i in images if i not in i_finish]
        print(f"Generate {data_type}, remaining samples: {len(images)}")

        generator = torch.Generator(device="cuda").manual_seed(0)

        for mask_path, image_path in zip(masks, images):
            file_name = os.path.basename(image_path)

            with torch.no_grad():
                image = Image.open(image_path)
                mask = Image.open(mask_path).convert("L")
                image = resize_img(image)
                w, h = image.size
                mask = mask.resize((w, h), Image.NEAREST)
                mask = mask.point(lambda p: 255 if p >= 128 else 0)

                prompt = "Question: What is the object in the image? Answer:"
                mask_inv = ImageOps.invert(mask)
                mask_object = Image.composite(image, Image.new(mode="RGB", size=(w, h), color=(255, 255, 255)), mask_inv)

                inputs = processor(images=mask_object, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
                generated_ids = blip_model.generate(**inputs, max_new_tokens=20)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                cat = generated_text.replace("Question: What is the object in the image? Answer:", "")\
                    .replace("It is ", "").replace("it is ", "").replace("It's ", "").replace("The object is ", "")
                print(image_path, generated_text, cat)

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
                    shuffle_image1 = Image.composite(image, shuffle_image, mask_inv)
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
                    num_inference_steps=30,
                    seed=42,
                    width=w,
                    height=h
                )
                imgname = os.path.basename(image_path).split('.')[0]
                generated_image[0].save(os.path.join(out_dir, f"{imgname}.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blip2_model_path", type=str, default="Salesforce/blip2-opt-6.7b")
    parser.add_argument("--inpaint_model_path", type=str, default="AI-ModelScope/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--ip_ckpt", type=str, required=True, help="path to ip_adapter.bin checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True, help="root of LAKERED_DATASET")
    parser.add_argument("--dst_root", type=str, required=True, help="output save root")
    parser.add_argument("--shuffle", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
