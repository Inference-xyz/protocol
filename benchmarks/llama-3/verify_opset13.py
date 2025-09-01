# verify_opset13.py
import onnx, onnxruntime as ort

# Check the model
onnx.checker.check_model("model_opset13_static.onnx")
print("✓ Model validation passed")

# Test with ONNX Runtime
sess = ort.InferenceSession("model_opset13_static.onnx", providers=["CPUExecutionProvider"])
print("✓ ONNX Runtime session created successfully")

print("Inputs:")
for i in sess.get_inputs():
    print(f"  {i.name}: shape={i.shape}, type={i.type}")

print("Outputs:")
for o in sess.get_outputs():
    print(f"  {o.name}: shape={o.shape}, type={o.type}")

print("\nModel is ready for ezkl testing!")
