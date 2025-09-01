from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="google/gemma-3-270m",
    output="onnx_models/gemma-3-270m",
    task="causal-lm",
    opset=17,
    device="cpu",
    cache_dir="hf_models/gemma-3-270m",
    pad_token_id=None,
    trust_remote_code=True,
    no_post_process=False,
    _from_auto=False,
    use_auth_token=True
)