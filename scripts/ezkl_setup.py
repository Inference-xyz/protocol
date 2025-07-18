import os
import subprocess
import torch
import torch.nn as nn
import ezkl
import asyncio

WORKDIR = "ezkl_demo"
SRS_PATH = os.path.expanduser("~/.ezkl/srs/kzg17.srs")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 4)
        self.l2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))

def download_srs():
    if not os.path.exists(SRS_PATH):
        print("📥 Downloading SRS...")
        subprocess.run([
            "ezkl", "get-srs",
            "--settings-path", "settings.json"
        ], check=True)
        print("✅ SRS downloaded")

def main():
    os.makedirs(WORKDIR, exist_ok=True)
    os.chdir(WORKDIR)

    model = SimpleModel()
    with torch.no_grad():
        model.l1.weight[:] = torch.tensor([[0.5, 0.3], [0.2, 0.8], [0.1, 0.9], [0.7, 0.4]])
        model.l1.bias[:] = torch.tensor([0.1, 0.2, 0.3, 0.4])
        model.l2.weight[:] = torch.tensor([[0.25] * 4])
        model.l2.bias[:] = torch.tensor([0.0])

    dummy_input = torch.randn(1, 2)
    torch.onnx.export(
        model, dummy_input, "model.onnx",
        export_params=True, opset_version=17,
        input_names=["input"], output_names=["output"]
    )
    print("✅ model.onnx exported")

    ezkl.gen_settings("model.onnx", "settings.json")
    print("✅ settings.json generated")

    download_srs()

    ezkl.compile_circuit("model.onnx", "network.ezkl", "settings.json")
    print("✅ circuit compiled")

    ezkl.setup("network.ezkl", "vk.key", "pk.key", SRS_PATH)
    print("✅ proving + verifying keys generated")

    async def create_verifier():
        ezkl.create_evm_verifier("vk.key", "settings.json", "verifier.sol", srs_path=SRS_PATH)
    asyncio.run(create_verifier())

    print("✅ verifier.sol generated")

if __name__ == "__main__":
    main()
