import onnx

m = onnx.load("model.onnx")
ops = sorted({(n.domain or "ai.onnx", n.op_type) for n in m.graph.node})
for dom, op in ops:
    print(f"{dom}::{op}")