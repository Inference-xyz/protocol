# make_inputs.py
import json, argparse, os

def build_zeros_kv(batch: int, heads: int, past: int, head_dim: int):
    # shape: [batch, heads, past, head_dim]
    return [
        [
            [
                [0.0 for _ in range(head_dim)]
                for _ in range(past)
            ]
            for _ in range(heads)
        ]
        for _ in range(batch)
    ]

def main():
    ap = argparse.ArgumentParser(description="Generate ezkl input.json for Llama-3.2-1B (ONNX with_past).")
    ap.add_argument("--token", type=int, default=1, help="Current input token id (default: 1)")
    ap.add_argument("--past", type=int, default=1, help="past_sequence_length P (default: 1; avoid 0 for safety)")
    ap.add_argument("--batch", type=int, default=1, help="batch size (must match ONNX; default: 1)")
    ap.add_argument("--seq", type=int, default=1, help="sequence_length for current step (default: 1)")
    ap.add_argument("--layers", type=int, default=16, help="number of transformer layers with KV (default: 16)")
    ap.add_argument("--heads", type=int, default=8, help="attention heads (default: 8)")
    ap.add_argument("--head-dim", type=int, default=64, help="head dimension (default: 64)")
    ap.add_argument("--out", type=str, default="ezkl_proof/input.json", help="output JSON path")
    args = ap.parse_args()

    # Basic shapes inferred from your ONNX:
    B = args.batch
    S = args.seq
    P = args.past
    H = args.heads
    D = args.head_dim
    L = args.layers

    # Sanity
    assert B == 1, "This script assumes batch=1 for ezkl PoC"
    assert S == 1, "This script assumes sequence_length=1 per generation step"

    data = {
        # [B, S]
        "input_ids": [[int(args.token)] for _ in range(B)],
        # [B, P+1]
        "attention_mask": [[1]*(P+1) for _ in range(B)],
        # [B, S] ; typical position id for this step is P (0-based positions)
        "position_ids": [[P] for _ in range(B)],
    }

    # Add past_key_values.{i}.key/value for i in [0..L-1], each [B,H,P,D]
    zeros = build_zeros_kv(B, H, P, D)
    for i in range(L):
        data[f"past_key_values.{i}.key"]   = zeros
        data[f"past_key_values.{i}.value"] = zeros

    # Ensure dir exists
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out}")
    print(f"- input_ids shape: [{B},{S}]")
    print(f"- attention_mask shape: [{B},{P+1}]")
    print(f"- position_ids shape: [{B},{S}]")
    print(f"- past_key_values.* shape: [{B},{H},{P},{D}] x {L} layers")

if __name__ == "__main__":
    main()