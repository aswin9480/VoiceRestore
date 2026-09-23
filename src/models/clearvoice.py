from pathlib import Path


class ClearVoiceAdapter:
    def __init__(self, device: str, cache: Path, options: dict):
        import os

        from clearvoice import ClearVoice
        from clearvoice.network_wrapper import network_wrapper
        from clearvoice.networks import SpeechModel

        checkpoint = cache / "clearvoice" / "MossFormer2_SE_48K"
        manifest = checkpoint / "last_best_checkpoint"

        def complete():
            if not manifest.is_file():
                return False
            names = [name.strip() for name in manifest.read_text(encoding="utf-8").splitlines() if name.strip()]
            return bool(names) and all(Path(name).name == name and (checkpoint / name).is_file() for name in names)

        if not complete():
            if os.environ.get("HF_HUB_OFFLINE") == "1":
                raise FileNotFoundError("ClearVoice weights are missing. Use Download / Verify Models first.")
            from huggingface_hub import snapshot_download

            snapshot_download("alibabasglab/MossFormer2_SE_48K", local_dir=str(checkpoint))
        names = manifest.read_text(encoding="utf-8").splitlines()
        if not complete():
            raise FileNotFoundError("ClearVoice checkpoint is incomplete. Remove its cache folder and download again.")
        original_args = network_wrapper.load_args_se

        def load_args(wrapper):
            original_args(wrapper)
            wrapper.args.checkpoint_dir = str(checkpoint)
            wrapper.args.use_cuda = int(device == "cuda")

        def strict_load(model_self, model, checkpoint_path, model_key=None):
            import torch

            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = state.get(model_key, state)
            mapped = {}
            for key, value in model.state_dict().items():
                match = next(
                    (
                        candidate
                        for candidate in [key, key.removeprefix("module."), "module." + key]
                        if candidate in state and state[candidate].shape == value.shape
                    ),
                    None,
                )
                if match is None:
                    raise RuntimeError(f"ClearVoice checkpoint is incompatible: missing {key}.")
                mapped[key] = state[match]
            model.load_state_dict(mapped, strict=True)

        network_wrapper.load_args_se = load_args
        SpeechModel._load_model = strict_load
        # CUDA visibility is configured before torch import in the isolated worker.
        try:
            self.model = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
        finally:
            network_wrapper.load_args_se = original_args
        from .downloads import fingerprint

        self.runtime = {"weights": [fingerprint(checkpoint / name.strip()) for name in names if name.strip()]}

    def process(self, source: Path, target: Path, options: dict) -> None:
        result = self.model(input_path=str(source), online_write=False)
        self.model.write(result, output_path=str(target))
