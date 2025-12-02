#!/usr/bin/env python3
"""
Run YOLOv8 inference on single image
"""

import argparse
import sys
import time
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference_pytorch import PyTorchInference
from src.inference_tensorrt import TensorRTInference
from src.visualize_detection import draw_detections, save_annotated_image


def run_inference(
    image_path: str,
    model_path: str = None,
    engine_path: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    output_path: str = None,
    show: bool = True
):
    """
    Run inference on single image
    
    Args:
        image_path: Path to input image
        model_path: Path to PyTorch model
        engine_path: Path to TensorRT engine
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
        output_path: Path to save output
        show: Display result
    """
    # Read image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to read image {image_path}")
        sys.exit(1)
    
    print(f"Image shape: {image.shape}")
    
    # Determine inference backend
    if engine_path and Path(engine_path).exists():
        print(f"\nUsing TensorRT engine: {engine_path}")
        try:
            engine = TensorRTInference(
                engine_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            backend = "TensorRT"
        except Exception as e:
            print(f"Error loading TensorRT engine: {e}")
            print("Falling back to PyTorch")
            engine = None
    else:
        engine = None
    
    if engine is None:
        if model_path and Path(model_path).exists():
            print(f"\nUsing PyTorch model: {model_path}")
            engine = PyTorchInference(
                model_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            backend = "PyTorch"
        else:
            print("Error: No valid model or engine provided")
            sys.exit(1)
    
    # Run inference
    print("\nRunning inference...")
    start_time = time.perf_counter()
    
    detections, info = engine.infer(image, return_preprocessed=True)
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    # Print results
    print(f"\nInference Results:")
    print(f"  Backend: {backend}")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Inference time: {info['inference_time']:.2f} ms")
    print(f"  Preprocessing time: {total_time - info['inference_time']:.2f} ms")
    print(f"  Number of detections: {len(detections)}")
    
    if len(detections) > 0:
        print("\nDetections:")
        from src.postprocessing import get_class_name
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            class_name = get_class_name(int(cls))
            print(f"  {i+1}. {class_name}: {conf:.3f} - Box: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
    
    # Visualize results
    annotated = draw_detections(image, detections)
    
    # Add performance info
    fps = 1000 / total_time
    cv2.putText(
        annotated,
        f"{backend} - FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # Save output
    if output_path:
        save_annotated_image(image, detections, output_path)
        print(f"\nSaved output to: {output_path}")
    
    # Display result
    if show:
        print("\nPress any key to close...")
        cv2.imshow("YOLOv8 Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference on image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='Path to PyTorch model')
    parser.add_argument('--engine', type=str,
                        help='Path to TensorRT engine')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--output', type=str,
                        help='Path to save output image')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display result')
    
    args = parser.parse_args()
    
    # Check input image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Run inference
    run_inference(
        image_path=args.image,
        model_path=args.model,
        engine_path=args.engine,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        output_path=args.output,
        show=not args.no_show
    )


if __name__ == '__main__':
    main()