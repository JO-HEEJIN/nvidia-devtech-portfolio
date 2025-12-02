# YOLOv8 TensorRT Model

This directory should contain the YOLOv8 TensorRT engine file.

## Model Requirements

- **Filename**: `model.plan` (TensorRT engine format)
- **Input**: Images (3x640x640) normalized to [0, 1]
- **Output**: Detections (Nx84) where N is number of boxes

## Generate TensorRT Engine

```bash
# Export YOLOv8 to ONNX first
python -c "
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.export(format='onnx', imgsz=640, dynamic=True)
"

# Build TensorRT engine
trtexec --onnx=yolov8s.onnx \
        --saveEngine=model.plan \
        --fp16 \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:4x3x640x640 \
        --maxShapes=images:8x3x640x640
```

## Input Preprocessing

```python
import cv2
import numpy as np

def preprocess_image(image_path):
    # Read and resize image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    
    # Convert BGR to RGB and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # HWC to CHW format
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img
```

## Output Format

- **Shape**: [batch_size, num_detections, 84]
- **Content**: [x, y, w, h, confidence, class_0_prob, ..., class_79_prob]
- **Classes**: 80 COCO object classes

## Expected Performance

- **Latency**: ~6ms (batch size 1, FP16)
- **Throughput**: ~185 QPS (batch size 8)
- **Memory**: ~1.2GB GPU memory
- **Accuracy**: mAP@0.5 = 44.9 on COCO