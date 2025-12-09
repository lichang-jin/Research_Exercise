import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F


def load_and_preprocess_image_square(image_path_list, target_size=1024):
    """Load and preprocess images by center padding to square and resizing to target size."""
    if len(image_path_list) == 0:
        raise ValueError("The list of images is empty.")

    images = []
    original_coords = []
    image_transforms = transforms.ToTensor()
    for image_path in image_path_list:
        image = Image.open(image_path)
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
        width, height = image.size
        max_dim = max(width, height)
        left, top = (max_dim - width) // 2, (max_dim - height) // 2
        scale = target_size / max_dim
        x1, x2, y1, y2 = left, left + width, top, top + height
        original_coords.append(np.array([x1 * scale, y1 * scale, x2 * scale, y2 * scale, width, height]))
        square_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_image.paste(image, (left, top))
        square_image = square_image.resize((target_size, target_size), Image.Resampling.BICUBIC)
        square_image = image_transforms(square_image)
        images.append(square_image)

    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)
    return images, original_coords


def load_and_preprocess_image(image_path_list, mode="crop", target_size=518):
    """A quick start function to load and preprocess images for model input."""
    if len(image_path_list) == 0:
        raise ValueError("The list of images is empty.")

    images = []
    shapes = set()
    image_transforms = transforms.ToTensor()
    for image_path in image_path_list:
        image = Image.open(image_path)
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
        width, height = image.size
        shapes.add((width, height))
        if mode == "crop":
            new_width = target_size
            new_height = round(height * (target_size / width) / 14) * 14
        elif mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (target_size / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (target_size / height) / 14) * 14
        else:
            raise ValueError(f"Invalid mode {mode}. Use 'crop' or 'pad'.")

        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        image = image_transforms(image)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            image = image[:, start_y : start_y + target_size, :]
        elif mode == "pad":
            h_padding = target_size - image.shape[1]
            w_padding = target_size - image.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top, pad_left = h_padding // 2, w_padding // 2
                pad_bottom, pad_right = h_padding - pad_top, w_padding - pad_left
                image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
        shapes.add((image.shape[1], image.shape[2]))
        images.append(image)

    if len(shapes) > 1:
        print(f"WARNING: The images have different shapes: {shapes}.")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded_images = []
        for image in images:
            h_padding = max_height - image.shape[1]
            w_padding = max_width - image.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top, pad_left = h_padding // 2, w_padding // 2
                pad_bottom, pad_right = h_padding - pad_top, w_padding - pad_left
                image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
            padded_images.append(image)
        images = padded_images

    images = torch.stack(images)
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
    return images


