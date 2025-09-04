#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Gemma-3-270M with ONLY the first transformer layer (after-embeddings) and post-process ONNX:
- Input:  inputs_embeds [B, T, H], attention_mask [B, T]
- Output: layer1_hidden [B, T, H]
- Replaces norms in the first block with primitive RMSNorm
- Scales inputs inside the graph (helps fixed-point backends)
- Clips large constants in ONNX initializers/Constant nodes (avoid EZKL truncation)
- Opset 18, fixed axes
"""

import os
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

import onnx
import numpy as np
from onnx import numpy_helper


# ---------- Primitive RMSNorm (no fused LayerNorm in ONNX) ----------
class RMSNormExport(nn.Module):
    """RMSNorm using primitive ops to avoid fused LayerNorm in ONNX."""
    def __init__(self, w: torch.Tensor, eps: float = 1e-6, b: torch.Tensor = None):
        super().__init__()
        self.weight = nn.Parameter(w.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(b.detach().clone(), requires_grad=False) if b is not None else None
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = torch.mean(x * x, dim=-1, keepdim=True)
        y = x * torch.rsqrt(ms + self.eps)
        y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


# ---------- Helpers to locate and trim layers ----------
def find_layers(model: nn.Module):
    """Locate transformer ModuleList (Gemma/LLaMA-style)."""
    for path in [("model", "layers"), ("transformer", "layers"), ("model", "decoder", "layers")]:
        obj = model
        ok = True
        for p in path:
            if not hasattr(obj, p):
                ok = False; break
            obj = getattr(obj, p)
        if ok and isinstance(obj, nn.ModuleList):
            owner = model
            for p in path[:-1]:
                owner = getattr(owner, p)
            return owner, path[-1], obj
    raise RuntimeError("Could not find transformer layers ModuleList")


def trim_to_first_layer(model: nn.Module):
    """Keep only layer[0] and update config."""
    owner, attr, layers = find_layers(model)
    if len(layers) < 1:
        raise RuntimeError("Model has no layers")
    setattr(owner, attr, nn.ModuleList([layers[0]]))
    if hasattr(model, "config"):
        for k in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(model.config, k):
                setattr(model.config, k, 1)
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"  # simpler attention path


def replace_norms_in_first(model: nn.Module):
    """Replace norms in the first layer with RMSNormExport."""
    owner, attr, layers = find_layers(model)
    blk = getattr(owner, attr)[0]
    for name, subm in list(blk.named_modules()):
        is_ln = isinstance(subm, nn.LayerNorm)
        is_rms_like = (hasattr(subm, "weight") and hasattr(subm, "eps") and not is_ln)
        if not (is_ln or is_rms_like):
            continue
        w = subm.weight
        b = getattr(subm, "bias", None)
        eps = float(getattr(subm, "eps", 1e-6))
        repl = RMSNormExport(w, eps=eps, b=b)
        parent = blk
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], repl)


# ---------- Wrapper that starts after embeddings ----------
class FirstLayerAfterEmbeds(nn.Module):
    """Takes inputs_embeds and returns first-layer hidden. Scales inputs to help fixed-point."""
    def __init__(self, base: nn.Module, in_scale: float = 0.25):
        super().__init__()
        self.m = base
        self.s = float(in_scale)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        x = inputs_embeds * self.s
        out = self.m(inputs_embeds=x, attention_mask=attention_mask, use_cache=False, output_hidden_states=True)
        return out.hidden_states[1]  # first layer hidden


# ---------- ONNX post-processing: clip large constants ----------
def clip_large_constants(input_path: str, output_path: str, clip: float = 1e4):
    """
    Clip large floating constants in initializers and Constant nodes to [-clip, clip].
    This prevents fixed-point overflow in backends like EZKL.
    """
    m = onnx.load(input_path)

    # 1) Initializers
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype.kind == "f":
            if np.isfinite(arr).any():
                maxabs = float(np.max(np.abs(arr)))
                if maxabs > clip:
                    arr = np.clip(arr, -clip, clip).astype(arr.dtype)
                    init.CopyFrom(numpy_helper.from_array(arr, init.name))

    # 2) Constant nodes
    for node in m.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    arr = numpy_helper.to_array(attr.t)
                    if arr.dtype.kind == "f":
                        if np.isfinite(arr).any():
                            maxabs = float(np.max(np.abs(arr)))
                            if maxabs > clip:
                                arr = np.clip(arr, -clip, clip).astype(arr.dtype)
                                attr.t.CopyFrom(numpy_helper.from_array(arr))

    onnx.save(m, output_path)
    print(f"[OK] Clipped ONNX saved: {output_path} (clip=±{clip:g})")


# ---------- Load, trim, export ----------
def load_model_tok(model_id: str, hf_token: str = None):
    """Load tokenizer (slow) and model on CPU in fp32."""
    if hf_token:
        try:
            login(token=hf_token)
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map=None, trust_remote_code=True
    ).cpu().eval()

    if hasattr(mdl.config, "_attn_implementation"):
        mdl.config._attn_implementation = "eager"

    return mdl, tok


def export_first_layer_after_embeds(model_id: str,
                                    out_path: str,
                                    seq_len: int,
                                    in_scale: float,
                                    hf_token: str = None):
    """Export ONNX that starts after embeddings and returns first-layer hidden."""
    mdl, tok = load_model_tok(model_id, hf_token)
    trim_to_first_layer(mdl)
    replace_norms_in_first(mdl)

    enc = tok("hello", return_tensors="pt", max_length=seq_len, padding="max_length", truncation=True)
    input_ids = enc["input_ids"]
    if tok.pad_token_id is None:
        # ensure we have a pad id for mask creation
        tok.pad_token = tok.eos_token or tok.unk_token or tok.cls_token
    attn = (input_ids != tok.pad_token_id).long()

    with torch.no_grad():
        emb = mdl.get_input_embeddings()(input_ids)  # [1, T, H]

    wrapper = FirstLayerAfterEmbeds(mdl, in_scale=in_scale)

    print(f"[+] Exporting ONNX → {out_path}")
    torch.onnx.export(
        wrapper,
        (emb, attn),
        out_path,
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["layer1_hidden"],
        opset_version=18,
        dynamic_axes=None,  # fixed axes
    )
    print(f"[OK] ONNX export complete: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-3-270m", help="HF model id")
    ap.add_argument("--out-onnx", default="gemma3_270m_first1_after_embeds.onnx", help="raw ONNX path")
    ap.add_argument("--out-onnx-clipped", default="gemma3_270m_first1_after_embeds_clipped.onnx",
                    help="clipped ONNX path (for EZKL)")
    ap.add_argument("--seq-len", type=int, default=64, help="sequence length for dummy input")
    ap.add_argument("--in-scale", type=float, default=0.25, help="input downscale factor inside the graph")
    ap.add_argument("--clip-abs", type=float, default=1e4, help="abs value to clip ONNX constants to")
    ap.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Hugging Face token")
    args = ap.parse_args()

    export_first_layer_after_embeds(args.model_id, args.out_onnx, args.seq_len, args.in_scale, args.hf_token)
    clip_large_constants(args.out_onnx, args.out_onnx_clipped, clip=args.clip_abs)


if __name__ == "__main__":
    main()