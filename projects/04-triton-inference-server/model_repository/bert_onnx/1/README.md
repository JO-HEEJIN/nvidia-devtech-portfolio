# BERT ONNX Model

This directory should contain the BERT ONNX model file.

## Model Requirements

- **Filename**: `model.onnx` (ONNX format)
- **Task**: Sequence classification (sentiment analysis)
- **Max Sequence Length**: 128 tokens
- **Classes**: 2 (negative/positive)

## Export BERT to ONNX

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pretrained model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create dummy inputs
dummy_text = "This is a test sentence."
inputs = tokenizer(
    dummy_text,
    return_tensors="pt",
    max_length=128,
    padding="max_length",
    truncation=True
)

# Export to ONNX
torch.onnx.export(
    model,
    tuple(inputs.values()),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=12
)
```

## Input Preprocessing

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(text):
    # Tokenize text
    inputs = tokenizer(
        text,
        return_tensors="np",
        max_length=128,
        padding="max_length",
        truncation=True
    )
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
```

## Output Processing

```python
import numpy as np

def postprocess_logits(logits):
    # Apply softmax to get probabilities
    probabilities = np.softmax(logits, axis=-1)
    
    # Get predicted class
    predicted_class = np.argmax(probabilities, axis=-1)
    
    # Map to labels
    labels = ["negative", "positive"]
    predicted_label = labels[predicted_class[0]]
    
    return {
        "label": predicted_label,
        "confidence": float(probabilities[0][predicted_class[0]])
    }
```

## Expected Performance

- **Latency**: ~8ms (batch size 1)
- **Throughput**: ~320 QPS (batch size 16)
- **Memory**: ~1.5GB GPU memory
- **Accuracy**: ~91% on IMDB sentiment dataset