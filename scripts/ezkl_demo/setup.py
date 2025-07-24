import torch
import torch.nn as nn
import numpy as np

# Create a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(4, 2)
        
    def forward(self, x):
        return torch.relu(self.linear(x))

# Generate model and input
model = SimpleModel()
dummy_input = torch.randn(1, 4)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

# Save input tensor
torch.save(dummy_input, "input.pt")

print("✅ Model and input files generated")
print("- model.onnx: ONNX model file")
print("- input.pt: Input tensor file") 