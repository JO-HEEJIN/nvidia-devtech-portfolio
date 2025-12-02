# Sample Images Directory

This directory should contain sample images for testing YOLOv8 object detection.

## Recommended Test Images

Please add images with various scenarios:

1. **Street scenes** - Cars, pedestrians, traffic lights
2. **Indoor scenes** - People, furniture, electronics
3. **Nature scenes** - Animals, outdoor objects
4. **Crowded scenes** - Multiple overlapping objects
5. **Night/Low-light** - Challenging lighting conditions

## Image Requirements

- Format: JPEG or PNG
- Resolution: Any (will be resized to 640x640 for inference)
- Naming: Use descriptive names (e.g., street_scene.jpg, indoor_office.png)

## Download Sample Images

You can download COCO dataset sample images:
```bash
# Download sample COCO images
wget http://images.cocodataset.org/val2017/000000000139.jpg -O street.jpg
wget http://images.cocodataset.org/val2017/000000000285.jpg -O indoor.jpg
wget http://images.cocodataset.org/val2017/000000000632.jpg -O crowd.jpg
```

Or use any images from:
- Your own camera/phone
- Public datasets (COCO, Pascal VOC, Open Images)
- Stock photo websites (with appropriate licenses)

## Testing

Test with a single image:
```bash
python ../demo/run_image.py --image street.jpg --engine ../models/yolov8s.engine
```

Batch testing:
```bash
for img in *.jpg; do
    python ../demo/run_image.py --image $img --output ../results/${img%.jpg}_detected.jpg
done
```