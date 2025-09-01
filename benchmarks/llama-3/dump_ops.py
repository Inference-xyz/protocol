# export_llama_eager13.py
import os, torch, onnx
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B"
B, S = 1, 16

os.environ["HF_USE_FLASH_ATTENTION_2"] = "0"
os.environ["HF_USE_SDP_ATTENTION"] = "0"

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True
)
model.eval()

model.config.use_cache = False
model.config._attn_implementation = "eager"   # ("eager" / "math")

input_ids = torch.ones((B, S), dtype=torch.long)
attention_mask = torch.zeros((B, S), dtype=torch.long); attention_mask[:, :4] = 1

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "llama_eager13.onnx",
    opset_version=13,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes=None,
    do_constant_folding=False,
    training=torch.onnx.TrainingMode.EVAL,
)

onnx.checker.check_model("llama_eager13.onnx")
print("Saved llama_eager13.onnx")