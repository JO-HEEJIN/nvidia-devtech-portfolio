#!/usr/bin/env python3
"""
Real-time YOLOv8 inference on webcam
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from collections import deque

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference_pytorch import PyTorchInference
from src.inference_tensorrt import TensorRTInference
from src.visualize_detection import draw_detections, overlay_fps, create_video_writer


class WebcamDetection:
    """
    Real-time object detection on webcam stream
    """
    
    def __init__(
        self,
        model_path: str = None,
        engine_path: str = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        camera_id: int = 0,
        output_video: str = None
    ):
        """
        Initialize webcam detection
        
        Args:
            model_path: Path to PyTorch model
            engine_path: Path to TensorRT engine
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            camera_id: Camera device ID
            output_video: Path to save output video
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.camera_id = camera_id
        self.output_video = output_video
        
        # Setup inference engine
        self.setup_engine(model_path, engine_path)
        
        # Setup camera
        self.setup_camera()
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        
        # Video writer
        self.video_writer = None
        if output_video:
            self.setup_video_writer()
    
    def setup_engine(self, model_path, engine_path):
        """
        Setup inference engine
        """
        if engine_path and Path(engine_path).exists():
            print(f"Using TensorRT engine: {engine_path}")
            try:
                self.engine = TensorRTInference(
                    engine_path,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold
                )
                self.backend = "TensorRT"
                return
            except Exception as e:
                print(f"Error loading TensorRT: {e}")
        
        if model_path and Path(model_path).exists():
            print(f"Using PyTorch model: {model_path}")
            self.engine = PyTorchInference(
                model_path,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold
            )
            self.backend = "PyTorch"
        else:
            raise ValueError("No valid model or engine provided")
    
    def setup_camera(self):
        """
        Setup webcam capture
        """
        print(f"Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera opened: {self.width}x{self.height} @ {self.fps} FPS")
    
    def setup_video_writer(self):
        """
        Setup video writer for recording
        """
        self.video_writer = create_video_writer(
            self.output_video,
            fps=self.fps,
            frame_size=(self.width, self.height),
            codec='mp4v'
        )
        print(f"Recording to: {self.output_video}")
    
    def process_frame(self, frame):
        """
        Process single frame
        
        Args:
            frame: Input frame
        
        Returns:
            Annotated frame, detections, inference time
        """
        start_time = time.perf_counter()
        
        # Run inference
        detections, _ = self.engine.infer(frame, return_preprocessed=False)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Draw detections
        annotated = draw_detections(frame, detections)
        
        return annotated, detections, inference_time
    
    def run(self):
        """
        Main detection loop
        """
        print("\n" + "="*50)
        print("Starting real-time detection")
        print(f"Backend: {self.backend}")
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        print("Press 'r' to toggle recording")
        print("="*50 + "\n")
        
        recording = self.video_writer is not None
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process frame
                annotated, detections, inference_time = self.process_frame(frame)
                
                # Update FPS
                self.fps_history.append(1000 / inference_time)
                avg_fps = np.mean(self.fps_history)
                
                # Overlay information
                annotated = overlay_fps(annotated, avg_fps)
                
                # Add detection count
                cv2.putText(
                    annotated,
                    f"Detections: {len(detections)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Add backend info
                cv2.putText(
                    annotated,
                    f"{self.backend}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
                
                # Recording indicator
                if recording:
                    cv2.circle(annotated, (self.width - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(
                        annotated,
                        "REC",
                        (self.width - 80, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    self.video_writer.write(annotated)
                
                # Display frame
                cv2.imshow("YOLOv8 Webcam Detection", annotated)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, annotated)
                    print(f"Screenshot saved: {screenshot_path}")
                elif key == ord('r'):
                    # Toggle recording
                    if self.video_writer is None:
                        self.setup_video_writer()
                        recording = True
                        print("Recording started")
                    else:
                        recording = not recording
                        status = "resumed" if recording else "paused"
                        print(f"Recording {status}")
                
                frame_count += 1
                
                # Print statistics every 30 frames
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: {avg_fps:.1f} FPS, "
                          f"{len(detections)} detections, "
                          f"{inference_time:.1f} ms/frame")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Clean up resources
        """
        print("\nCleaning up...")
        
        if self.cap:
            self.cap.release()
        
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved to: {self.output_video}")
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Real-time YOLOv8 webcam detection')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='Path to PyTorch model')
    parser.add_argument('--engine', type=str,
                        help='Path to TensorRT engine')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--output-video', type=str,
                        help='Path to save output video')
    
    args = parser.parse_args()
    
    # Create detector
    try:
        detector = WebcamDetection(
            model_path=args.model,
            engine_path=args.engine,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            camera_id=args.camera,
            output_video=args.output_video
        )
        
        # Run detection
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()