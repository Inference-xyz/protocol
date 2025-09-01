
import argparse, os
import onnx
from onnx import helper, TensorProto, shape_inference, version_converter, ValueInfoProto
from onnx.external_data_helper import convert_model_to_external_data

def set_shape(vi: ValueInfoProto, dims):
    tt = vi.type.tensor_type
    tt.shape.dim.clear()
    for d in dims:
        dim = tt.shape.dim.add()
        dim.dim_value = int(d)

def find_vi(model, name):
    g = model.graph
    for coll in (g.input, g.output, g.value_info):
        for vi in coll:
            if vi.name == name:
                return vi
    return None

def ensure_input(model, name, elem_type, dims):
    vi = find_vi(model, name)
    if vi is None:
        vi = helper.make_tensor_value_info(name, elem_type, dims)
        model.graph.input.append(vi)
    else:
        set_shape(vi, dims)
    return vi

def ensure_output(model, name, elem_type, dims):
    vi = find_vi(model, name)
    if vi is None:
        vi = helper.make_tensor_value_info(name, elem_type, dims)
        model.graph.output.append(vi)
    else:
        set_shape(vi, dims)
    return vi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default="model_static13.onnx")
    ap.add_argument("--past", type=int, default=1)
    ap.add_argument("--seq", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=128256)
    ap.add_argument("--external", action="store_true")
    args = ap.parse_args()

    B, S, P, H, D, L, V = args.batch, args.seq, args.past, args.heads, args.head_dim, args.layers, args.vocab

    m = onnx.load(args.inp)

    if not m.opset_import or len(m.opset_import) == 0:
        m.opset_import.extend([helper.make_operatorsetid("", 13)])
    else:
        highest = max(x.version for x in m.opset_import if x.domain in ("", "ai.onnx"))
        if highest != 13:
            m = version_converter.convert_version(m, 13)

    ensure_input(m, "input_ids",      TensorProto.INT64, [B, S])
    ensure_input(m, "attention_mask", TensorProto.INT64, [B, P + S])
    ensure_input(m, "position_ids",   TensorProto.INT64, [B, S])

    for i in range(L):
        ensure_input(m, f"past_key_values.{i}.key",   TensorProto.FLOAT, [B, H, P, D])
        ensure_input(m, f"past_key_values.{i}.value", TensorProto.FLOAT, [B, H, P, D])

    ensure_output(m, "logits", TensorProto.FLOAT, [B, S, V])
    for i in range(L):
        ensure_output(m, f"present.{i}.key",   TensorProto.FLOAT, [B, H, P + S, D])
        ensure_output(m, f"present.{i}.value", TensorProto.FLOAT, [B, H, P + S, D])

    m = shape_inference.infer_shapes(m)

    if args.external:
        convert_model_to_external_data(
            m,
            all_tensors_to_one_file=True,
            location=os.path.splitext(args.out)[0] + ".onnx_data",
            size_threshold=1024,
            convert_attribute=False,
        )
    onnx.save_model(m, args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()