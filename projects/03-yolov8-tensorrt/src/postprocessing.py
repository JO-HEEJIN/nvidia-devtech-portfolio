"""
Postprocessing utilities for YOLOv8 detections
"""

import numpy as np
from typing import List, Tuple, Optional


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from center format to corner format
    
    Args:
        boxes: Boxes in [x_center, y_center, width, height] format
    
    Returns:
        Boxes in [x1, y1, x2, y2] format
    """
    output = np.copy(boxes)
    output[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
    output[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
    output[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
    output[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
    return output


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> np.ndarray:
    """
    Non-Maximum Suppression
    
    Args:
        boxes: Bounding boxes [N, 4] in xyxy format
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([])
    
    # Sort by score
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Take box with highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        ious = calculate_iou(current_box, other_boxes)
        
        # Remove boxes with high IoU
        indices = indices[1:][ious <= iou_threshold]
    
    return np.array(keep)


def calculate_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between one box and multiple boxes
    
    Args:
        box: Single box [4] in xyxy format
        boxes: Multiple boxes [N, 4] in xyxy format
    
    Returns:
        IoU values [N]
    """
    # Calculate intersection
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate union
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)
    
    return iou


def postprocess_predictions(
    predictions: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300
) -> List[np.ndarray]:
    """
    Postprocess YOLOv8 predictions
    
    Args:
        predictions: Raw model output [batch, num_boxes, 4 + 1 + num_classes]
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
        max_detections: Maximum detections per image
    
    Returns:
        List of detections per image, each [N, 6] with [x1, y1, x2, y2, conf, class]
    """
    batch_size = predictions.shape[0]
    results = []
    
    for i in range(batch_size):
        # Get predictions for this image
        pred = predictions[i]
        
        # YOLOv8 output format: [x, y, w, h, obj_conf, class_scores...]
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        
        # Get max class score and class id for each box
        if scores.shape[1] > 1:
            # Multi-class
            class_scores = scores[:, 1:]  # Skip objectness
            class_ids = np.argmax(class_scores, axis=1)
            confidences = scores[:, 0] * class_scores[np.arange(len(class_ids)), class_ids]
        else:
            # Single score (objectness only)
            confidences = scores[:, 0]
            class_ids = np.zeros(len(confidences), dtype=np.int32)
        
        # Filter by confidence
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            results.append(np.empty((0, 6)))
            continue
        
        # Convert to xyxy format
        boxes = xywh2xyxy(boxes)
        
        # Apply NMS per class
        keep_indices = []
        for cls in np.unique(class_ids):
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = confidences[cls_mask]
            
            # NMS for this class
            cls_keep = nms(cls_boxes, cls_scores, iou_threshold)
            
            # Get original indices
            original_indices = np.where(cls_mask)[0]
            keep_indices.extend(original_indices[cls_keep])
        
        keep_indices = np.array(keep_indices)
        
        # Limit detections
        if len(keep_indices) > max_detections:
            # Sort by confidence and keep top detections
            sorted_indices = np.argsort(confidences[keep_indices])[::-1]
            keep_indices = keep_indices[sorted_indices[:max_detections]]
        
        # Format output
        if len(keep_indices) > 0:
            detections = np.concatenate([
                boxes[keep_indices],
                confidences[keep_indices, np.newaxis],
                class_ids[keep_indices, np.newaxis].astype(np.float32)
            ], axis=1)
        else:
            detections = np.empty((0, 6))
        
        results.append(detections)
    
    return results


def filter_detections_by_class(
    detections: np.ndarray,
    target_classes: Optional[List[int]] = None
) -> np.ndarray:
    """
    Filter detections by class IDs
    
    Args:
        detections: Detections [N, 6] with [x1, y1, x2, y2, conf, class]
        target_classes: List of class IDs to keep (None keeps all)
    
    Returns:
        Filtered detections
    """
    if target_classes is None or len(detections) == 0:
        return detections
    
    class_ids = detections[:, 5].astype(int)
    mask = np.isin(class_ids, target_classes)
    
    return detections[mask]


def merge_batch_detections(
    batch_detections: List[np.ndarray]
) -> np.ndarray:
    """
    Merge detections from multiple images in a batch
    
    Args:
        batch_detections: List of detections per image
    
    Returns:
        Merged detections with added image index column
    """
    merged = []
    
    for img_idx, detections in enumerate(batch_detections):
        if len(detections) > 0:
            # Add image index as first column
            img_indices = np.full((len(detections), 1), img_idx)
            detections_with_idx = np.concatenate([img_indices, detections], axis=1)
            merged.append(detections_with_idx)
    
    if merged:
        return np.vstack(merged)
    else:
        return np.empty((0, 7))


def apply_confidence_threshold(
    detections: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Filter detections by confidence threshold
    
    Args:
        detections: Detections [N, 6] with [x1, y1, x2, y2, conf, class]
        threshold: Confidence threshold
    
    Returns:
        Filtered detections
    """
    if len(detections) == 0:
        return detections
    
    mask = detections[:, 4] >= threshold
    return detections[mask]


def scale_boxes_to_original(
    boxes: np.ndarray,
    scale_factor: float,
    padding: Tuple[int, int]
) -> np.ndarray:
    """
    Scale boxes back to original image coordinates
    
    Args:
        boxes: Boxes in [x1, y1, x2, y2] format
        scale_factor: Scale factor from preprocessing
        padding: Padding (top, left) from preprocessing
    
    Returns:
        Scaled boxes
    """
    # Remove padding
    boxes[:, [0, 2]] -= padding[1]  # x coordinates
    boxes[:, [1, 3]] -= padding[0]  # y coordinates
    
    # Scale back to original size
    boxes[:, :4] /= scale_factor
    
    return boxes


# COCO class names for visualization
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def get_class_name(class_id: int) -> str:
    """
    Get class name from class ID
    
    Args:
        class_id: Class ID
    
    Returns:
        Class name string
    """
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    else:
        return f'class_{class_id}'