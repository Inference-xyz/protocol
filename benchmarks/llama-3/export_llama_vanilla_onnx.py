import os
import argparse
import torch
import onnx
from onnx.external_data_helper import convert_model_to_external_data
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_USE_FLASH_ATTENTION_2"] = "0"
os.environ["HF_USE_SDP_ATTENTION"] = "0"

def make_simple_causal_mask(batch_size: int, seq_len: int, device=None, dtype=torch.float32):
    neg_inf = torch.finfo(torch.float32).min
    tri = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.float32), diagonal=1)
    mask = tri * neg_inf                    
    return mask.unsqueeze(0).unsqueeze(0)

def patch_masks(default_seq_len: int):
    import transformers.masking_utils as mu
    import transformers.models.llama.modeling_llama as ml

    def resolve_bs_S(*args, **kwargs):
        bs = None
        S  = None
        if len(args) >= 2:
            bs, S = args[0], args[1]
        else:
            bs = kwargs.get("batch_size") or kwargs.get("bs") or kwargs.get("B")
            S  = (kwargs.get("tgt_len") or kwargs.get("tgt_length") or
                  kwargs.get("seq_len") or kwargs.get("S") or kwargs.get("length"))
        if bs is None: bs = 1
        if S  is None: S  = default_seq_len
        return int(bs), int(S)

    def mask_interface_patched(*args, **kwargs):
        bs, S = resolve_bs_S(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype  = kwargs.get("dtype", torch.float32)
        return make_simple_causal_mask(bs, S, device=device, dtype=dtype)

    def create_causal_mask_patched(*args, **kwargs):
        bs, S = resolve_bs_S(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype  = kwargs.get("dtype", torch.float32)
        return make_simple_causal_mask(bs, S, device=device, dtype=dtype)

    mu.mask_interface = mask_interface_patched
    mu.create_causal_mask = create_causal_mask_patched
    ml.create_causal_mask = create_causal_mask_patched

def disable_sliding_window(model):
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for blk in model.model.layers:
            attn = getattr(blk, "self_attn", None)
            if attn is not None and hasattr(attn, "sliding_window"):
                attn.sliding_window = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--out", default="llama_vanilla13.onnx")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    _ = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    disable_sliding_window(model)

    patch_masks(default_seq_len=args.seq)

    B, S = 1, args.seq
    input_ids = torch.ones((B, S), dtype=torch.long)

    torch.onnx.export(
        model,
        input_ids,
        args.out,
        opset_version=args.opset,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=None,
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.EVAL,
    )

    m = onnx.load(args.out)
    convert_model_to_external_data(
        m,
        all_tensors_to_one_file=True,
        location=os.path.splitext(args.out)[0] + ".onnx_data",
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save_model(m, args.out)
    onnx.checker.check_model(args.out)
    print(f"Saved: {args.out}")
    print(f"External data: {os.path.splitext(args.out)[0]}.onnx_data")

if __name__ == "__main__":
    main()