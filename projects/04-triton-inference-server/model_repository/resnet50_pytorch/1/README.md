# ResNet50 PyTorch Model

This directory should contain the ResNet50 PyTorch model file.

## Model Requirements

- **Filename**: `model.pt` (TorchScript format)
- **Input**: Images (3x224x224) normalized with ImageNet statistics
- **Output**: 1000-class probabilities

## Generate Model

```python
import torch
import torchvision

# Load pretrained ResNet50
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# Create example input
example_input = torch.randn(1, 3, 224, 224)

# Export to TorchScript
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model.pt")
```

## Input Preprocessing

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## Expected Performance

- **Latency**: ~3ms (batch size 1)
- **Throughput**: ~280 QPS (batch size 8)
- **Memory**: ~200MB GPU memory
- **Accuracy**: 76.1% top-1 on ImageNet