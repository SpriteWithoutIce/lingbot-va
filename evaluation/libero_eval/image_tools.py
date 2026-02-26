"""
Image preprocessing for LIBERO evaluation.
Adapted from openpi_client/image_tools.
"""

import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float image to uint8 for network transfer."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
    return np.asarray(img, dtype=np.uint8)


def resize_with_pad(
    images: np.ndarray, height: int, width: int, method=Image.BILINEAR
) -> np.ndarray:
    """Resize image(s) to target size with padding (no distortion)."""
    if np.ndim(images) == 3:
        single = True
        images = images[np.newaxis, ...]
    else:
        single = False
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            np.array(
                _resize_with_pad_pil(
                    Image.fromarray(im.astype(np.uint8)), height, width, method=method
                )
            )
            for im in images
        ]
    )
    resized = resized.reshape(*original_shape[:-3], *resized.shape[-3:])
    return resized.squeeze(0) if single else resized


def _resize_with_pad_pil(
    image: Image.Image, height: int, width: int, method: int
) -> Image.Image:
    """Resize one image to target size with zero padding."""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)
    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, (height - resized_height) // 2)
    pad_width = max(0, (width - resized_width) // 2)
    zero_image.paste(resized_image, (pad_width, pad_height))
    return zero_image
