#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1) Load Gemma-3-270M from Hugging Face
2) Trim model to first two layers
3) Export to ONNX (opset 18, fixed axes)
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


def slice_first_two_layers_inplace(model) -> None:
    """Trim the transformer to the first two layers and update config."""
    possible_paths = [
        ("model", "layers"),
        ("transformer", "layers"),
        ("model", "decoder", "layers"),
    ]
    layers = None
    owner, attr = None, None

    for path in possible_paths:
        obj = model
        ok = True
        for p in path:
            if not hasattr(obj, p):
                ok = False
                break
            obj = getattr(obj, p)
        if ok and isinstance(obj, torch.nn.ModuleList):
            layers = obj
            owner = model
            for p in path[:-1]:
                owner = getattr(owner, p)
            attr = path[-1]
            break

    if layers is None:
        raise RuntimeError("Could not find ModuleList of layers. Check model structure or transformers version.")

    if len(layers) < 2:
        raise RuntimeError(f"Model has only {len(layers)} layers, need >= 2.")

    trimmed = torch.nn.ModuleList(list(layers)[:2])
    setattr(owner, attr, trimmed)

    if hasattr(model, "config"):
        for k in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(model.config, k):
                setattr(model.config, k, 2)
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"


class TwoLayerWrapperLogits(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.m = base
    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.logits


class TwoLayerWrapperWithHidden(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.m = base
    def forward(self, input_ids, attention_mask):
        out = self.m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True
        )
        return out.logits, out.hidden_states[1], out.hidden_states[2]


def load_model_and_tokenizer(model_id: str, hf_token: str = None):
    if hf_token:
        try:
            login(token=hf_token)
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    ).to("cpu").eval()

    return mdl, tok


def build_wrapper(model, with_hidden: bool):
    if with_hidden:
        wrapper = TwoLayerWrapperWithHidden(model)
        output_names = ["logits", "layer1_hidden", "layer2_hidden"]
    else:
        wrapper = TwoLayerWrapperLogits(model)
        output_names = ["logits"]
    return wrapper, output_names


def export_to_onnx(wrapper, tokenizer, out_path: str, seq_len: int, output_names):
    dummy = tokenizer("hello", return_tensors="pt", max_length=seq_len, padding="max_length", truncation=True)
    print(f"[+] Exporting ONNX → {out_path}")
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes=None,
        opset_version=18,
    )
    print("[✓] Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-3-270m", help="HF model id")
    parser.add_argument("--out-onnx", default="gemma3_270m_first2.onnx", help="output ONNX file")
    parser.add_argument("--seq-len", type=int, default=64, help="sequence length for dummy input")
    parser.add_argument("--with-hidden", action="store_true", help="export logits + first two hidden states")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Hugging Face token")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.hf_token)
    slice_first_two_layers_inplace(model)
    wrapper, output_names = build_wrapper(model, args.with_hidden)
    export_to_onnx(wrapper, tokenizer, args.out_onnx, args.seq_len, output_names)


if __name__ == "__main__":
    main()
