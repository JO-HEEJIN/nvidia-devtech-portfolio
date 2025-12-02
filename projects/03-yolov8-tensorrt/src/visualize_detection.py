"""
Visualization utilities for YOLOv8 detections
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from postprocessing import get_class_name


# Color palette for different classes (80 COCO classes)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (255, 128, 0), (128, 0, 255), (255, 0, 128),
    (0, 128, 255), (128, 128, 255), (128, 255, 128), (255, 128, 128),
    (128, 128, 128), (64, 128, 255), (64, 255, 128), (128, 64, 255),
    (255, 64, 128), (128, 255, 64), (255, 128, 64), (64, 64, 255),
    (64, 255, 64), (255, 64, 64), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (192, 192, 192), (64, 64, 64), (192, 64, 64), (64, 192, 64), (64, 64, 192),
    (192, 192, 64), (192, 64, 192), (64, 192, 192), (96, 96, 96), (160, 160, 160),
    (224, 224, 224), (32, 32, 32), (96, 32, 32), (32, 96, 32), (32, 32, 96),
    (96, 96, 32), (96, 32, 96), (32, 96, 96), (160, 32, 32), (32, 160, 32),
    (32, 32, 160), (160, 160, 32), (160, 32, 160), (32, 160, 160),
    (224, 32, 32), (32, 224, 32), (32, 32, 224), (224, 224, 32),
    (224, 32, 224), (32, 224, 224), (128, 96, 32), (128, 32, 96),
    (32, 128, 96), (96, 128, 32), (96, 32, 128), (32, 96, 128),
    (192, 96, 32), (192, 32, 96), (32, 192, 96), (96, 192, 32),
    (96, 32, 192), (32, 96, 192), (255, 96, 32), (255, 32, 96),
    (32, 255, 96), (96, 255, 32), (96, 32, 255), (32, 96, 255),
    (160, 128, 96), (160, 96, 128), (96, 160, 128), (128, 160, 96),
    (128, 96, 160), (96, 128, 160)
]


def draw_detections(
    image: np.ndarray,
    detections: np.ndarray,
    class_names: bool = True,
    confidence_scores: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1
) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image
        detections: Detections [N, 6] with [x1, y1, x2, y2, conf, class]
        class_names: Show class names
        confidence_scores: Show confidence scores
        box_thickness: Box line thickness
        font_scale: Font size scale
        font_thickness: Font line thickness
    
    Returns:
        Annotated image
    """
    # Create copy to avoid modifying original
    annotated = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        
        # Get color for this class
        color = COLORS[cls % len(COLORS)]
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
        
        # Prepare label
        if class_names and confidence_scores:
            label = f"{get_class_name(cls)}: {conf:.2f}"
        elif class_names:
            label = get_class_name(cls)
        elif confidence_scores:
            label = f"{conf:.2f}"
        else:
            label = ""
        
        if label:
            # Calculate label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw label background
            y_label = max(y1, label_h + 10)
            cv2.rectangle(
                annotated,
                (x1, y_label - label_h - 10),
                (x1 + label_w + 10, y_label + baseline - 10),
                color, -1
            )
            
            # Draw label text
            cv2.putText(
                annotated, label,
                (x1 + 5, y_label - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )
    
    return annotated


def save_annotated_image(
    image: np.ndarray,
    detections: np.ndarray,
    output_path: str,
    **kwargs
):
    """
    Save annotated image to file
    
    Args:
        image: Input image
        detections: Detections array
        output_path: Output file path
        **kwargs: Additional arguments for draw_detections
    """
    annotated = draw_detections(image, detections, **kwargs)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved annotated image to {output_path}")


def create_detection_grid(
    images: List[np.ndarray],
    detections_list: List[np.ndarray],
    grid_size: Tuple[int, int] = (2, 2),
    image_size: Tuple[int, int] = (640, 640)
) -> np.ndarray:
    """
    Create grid of detection results
    
    Args:
        images: List of images
        detections_list: List of detections per image
        grid_size: Grid dimensions (rows, cols)
        image_size: Size to resize each image
    
    Returns:
        Grid image
    """
    rows, cols = grid_size
    h, w = image_size
    
    # Create blank grid
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    # Fill grid
    for idx, (image, detections) in enumerate(zip(images, detections_list)):
        if idx >= rows * cols:
            break
        
        # Calculate position
        row = idx // cols
        col = idx % cols
        
        # Resize image
        resized = cv2.resize(image, (w, h))
        
        # Draw detections
        annotated = draw_detections(resized, detections)
        
        # Place in grid
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = annotated
    
    return grid


def overlay_fps(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Overlay FPS counter on image
    
    Args:
        image: Input image
        fps: FPS value
        position: Text position (x, y)
        font_scale: Font size
        color: Text color
        thickness: Text thickness
    
    Returns:
        Image with FPS overlay
    """
    text = f"FPS: {fps:.1f}"
    
    # Add background for better visibility
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    cv2.rectangle(
        image,
        (position[0] - 5, position[1] - text_h - 5),
        (position[0] + text_w + 5, position[1] + baseline + 5),
        (0, 0, 0), -1
    )
    
    cv2.putText(
        image, text, position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, thickness,
        cv2.LINE_AA
    )
    
    return image


def create_video_writer(
    output_path: str,
    fps: int = 30,
    frame_size: Tuple[int, int] = (1280, 720),
    codec: str = 'mp4v'
) -> cv2.VideoWriter:
    """
    Create video writer for saving detection videos
    
    Args:
        output_path: Output video path
        fps: Frames per second
        frame_size: Frame dimensions (width, height)
        codec: Video codec
    
    Returns:
        VideoWriter object
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, frame_size
    )
    
    return writer


def visualize_statistics(
    detections: np.ndarray,
    image_size: Tuple[int, int] = (640, 480)
) -> np.ndarray:
    """
    Create statistics visualization
    
    Args:
        detections: Detections array
        image_size: Output image size
    
    Returns:
        Statistics visualization image
    """
    # Create blank canvas
    canvas = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    
    if len(detections) == 0:
        cv2.putText(
            canvas, "No detections", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        return canvas
    
    # Count detections per class
    classes = detections[:, 5].astype(int)
    unique_classes, counts = np.unique(classes, return_counts=True)
    
    # Draw statistics
    y_offset = 50
    cv2.putText(
        canvas, f"Total detections: {len(detections)}", (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
    )
    
    y_offset += 40
    cv2.putText(
        canvas, "Detections by class:", (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
    )
    
    y_offset += 30
    for cls, count in zip(unique_classes, counts):
        class_name = get_class_name(cls)
        color = COLORS[cls % len(COLORS)]
        
        # Draw bar
        bar_length = int(count * 20)
        cv2.rectangle(
            canvas, (40, y_offset), (40 + bar_length, y_offset + 20),
            color, -1
        )
        
        # Draw text
        text = f"{class_name}: {count}"
        cv2.putText(
            canvas, text, (50 + bar_length, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
        
        y_offset += 30
        
        if y_offset > image_size[1] - 50:
            break
    
    # Draw confidence distribution
    confidences = detections[:, 4]
    avg_conf = np.mean(confidences)
    min_conf = np.min(confidences)
    max_conf = np.max(confidences)
    
    y_offset = image_size[1] - 100
    cv2.putText(
        canvas, f"Avg confidence: {avg_conf:.3f}", (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
    )
    
    y_offset += 25
    cv2.putText(
        canvas, f"Min: {min_conf:.3f}, Max: {max_conf:.3f}", (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
    )
    
    return canvas


def create_comparison_image(
    original: np.ndarray,
    pytorch_detections: np.ndarray,
    tensorrt_detections: np.ndarray,
    pytorch_time: float,
    tensorrt_time: float
) -> np.ndarray:
    """
    Create side-by-side comparison image
    
    Args:
        original: Original image
        pytorch_detections: PyTorch detections
        tensorrt_detections: TensorRT detections
        pytorch_time: PyTorch inference time (ms)
        tensorrt_time: TensorRT inference time (ms)
    
    Returns:
        Comparison image
    """
    h, w = original.shape[:2]
    
    # Create canvas for three images
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # Original image
    canvas[:, :w] = original
    cv2.putText(
        canvas, "Original", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    
    # PyTorch result
    pytorch_annotated = draw_detections(original, pytorch_detections)
    canvas[:, w:2*w] = pytorch_annotated
    cv2.putText(
        canvas, f"PyTorch: {pytorch_time:.1f}ms", (w + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    
    # TensorRT result
    tensorrt_annotated = draw_detections(original, tensorrt_detections)
    canvas[:, 2*w:] = tensorrt_annotated
    cv2.putText(
        canvas, f"TensorRT: {tensorrt_time:.1f}ms", (2*w + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    
    # Add speedup info
    speedup = pytorch_time / tensorrt_time
    cv2.putText(
        canvas, f"Speedup: {speedup:.1f}x", (2*w + 10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
    )
    
    return canvas