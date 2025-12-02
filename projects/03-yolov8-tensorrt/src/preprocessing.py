"""
Image preprocessing utilities for YOLOv8
"""

import cv2
import numpy as np
from typing import Tuple, List, Union


def letterbox(
    image: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Letterbox resize maintaining aspect ratio
    
    Args:
        image: Input image
        new_shape: Target shape (height, width) or single value for square
        color: Padding color
        auto: Automatic padding to stride multiple
        scaleFill: Stretch image to fill new shape
        scaleup: Allow scaling up
        stride: Stride for auto padding
    
    Returns:
        Resized image, scale ratio, padding (top/left, bottom/right)
    """
    # Get current shape
    shape = image.shape[:2]  # height, width
    
    # Convert new_shape to (height, width)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    
    # Compute new unpadded dimensions
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    # Compute padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    if auto:
        # Minimum padding to reach stride multiple
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        # Stretch to exact shape
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    # Divide padding into two sides
    dw /= 2
    dh /= 2
    
    # Resize image
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    
    return image, r, (top, left)


def preprocess_image(
    image: np.ndarray,
    input_size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
    bgr_to_rgb: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Preprocess image for YOLOv8 inference
    
    Args:
        image: Input image (BGR format from cv2)
        input_size: Model input size (height, width)
        normalize: Normalize to [0, 1]
        bgr_to_rgb: Convert BGR to RGB
    
    Returns:
        Preprocessed tensor, original image, scale ratio, padding
    """
    # Store original image
    original_image = image.copy()
    
    # Letterbox resize
    image, ratio, padding = letterbox(image, input_size)
    
    # Color space conversion
    if bgr_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    # HWC to CHW format
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Ensure contiguous array
    image = np.ascontiguousarray(image)
    
    return image, original_image, ratio, padding


def preprocess_batch(
    images: List[np.ndarray],
    input_size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
    bgr_to_rgb: bool = True
) -> Tuple[np.ndarray, List[np.ndarray], List[float], List[Tuple[int, int]]]:
    """
    Preprocess batch of images
    
    Args:
        images: List of input images
        input_size: Model input size
        normalize: Normalize to [0, 1]
        bgr_to_rgb: Convert BGR to RGB
    
    Returns:
        Batch tensor, original images, scale ratios, paddings
    """
    batch_tensor = []
    original_images = []
    ratios = []
    paddings = []
    
    for image in images:
        # Preprocess each image
        tensor, original, ratio, padding = preprocess_image(
            image, input_size, normalize, bgr_to_rgb
        )
        
        batch_tensor.append(tensor[0])  # Remove batch dimension
        original_images.append(original)
        ratios.append(ratio)
        paddings.append(padding)
    
    # Stack into batch
    batch_tensor = np.stack(batch_tensor, axis=0)
    
    return batch_tensor, original_images, ratios, paddings


def rescale_boxes(
    boxes: np.ndarray,
    original_shape: Tuple[int, int],
    ratio: float,
    padding: Tuple[int, int]
) -> np.ndarray:
    """
    Rescale bounding boxes to original image coordinates
    
    Args:
        boxes: Boxes in format [x1, y1, x2, y2]
        original_shape: Original image (height, width)
        ratio: Scale ratio from letterbox
        padding: Padding (top, left) from letterbox
    
    Returns:
        Rescaled boxes
    """
    # Remove padding
    boxes[:, [0, 2]] -= padding[1]  # x coordinates
    boxes[:, [1, 3]] -= padding[0]  # y coordinates
    
    # Rescale to original size
    boxes[:, :4] /= ratio
    
    # Clip to image bounds
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_shape[0])
    
    return boxes


def prepare_input_tensor(
    image_path: str,
    input_size: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Prepare input tensor from image path
    
    Args:
        image_path: Path to input image
        input_size: Model input size
    
    Returns:
        Input tensor, original image, scale ratio, padding
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Preprocess
    return preprocess_image(image, input_size)


def resize_and_pad(
    image: np.ndarray,
    target_size: int = 640
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Simple resize and pad for inference
    
    Args:
        image: Input image
        target_size: Target size for both dimensions
    
    Returns:
        Resized image, scale, padding
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Calculate padding
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # Add padding
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    return padded, scale, (top, left)