def torch_device(preference: str = "auto") -> dict:
    import torch

    available = torch.cuda.is_available()
    kind = "cuda" if preference != "cpu" and available else "cpu"
    if kind == "cuda":
        try:
            # Availability alone doesn't guarantee the installed wheel supports this GPU.
            torch.ones(1, device="cuda").sum().item()
        except RuntimeError:
            kind = "cpu"
    return {
        "type": kind,
        "gpu": torch.cuda.get_device_name(0) if kind == "cuda" else None,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "fallback": preference == "cuda" and kind == "cpu",
    }
