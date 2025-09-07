# EZKL Model Benchmarks

## Model Performance Table

| Model | Task | Parameters | Tag | Verified |
|-------|------|------------|------------------|----------|
| MobileNetV2 (224x224) | CV | ~3.4M | `google/mobilenet_v2_1.0_224` | ✅ |
| ResNet-18 | CV | ~11.7M | `microsoft/resnet-18` | ✅ |
| DistilGPT-2 | NLP (LM) | ~82M | `distilgpt2` | ✅ |
| GPT-2 (117M) | NLP (LM) | ~117M | `gpt2` | ✅ |
| DistilBERT | NLP (encoder) | ~66M | `distilbert-base-uncased` | ✅ |
| T5-small | NLP (seq2seq) | ~60M | `t5-small` | ✅ |
| ViT-tiny/16 | Vision Transformer | ~5-6M | `google/vit-base-patch16-224` | ✅ |

## Usage example

```bash
python download_model.py google/mobilenet_v2_1.0_224 -o models/mobilenet_v2

# Run benchmark
python benchmark.py --model <model.onnx> --input <input.json>
```
