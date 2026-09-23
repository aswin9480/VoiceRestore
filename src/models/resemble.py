from pathlib import Path


class ResembleAdapter:
    def __init__(self, device: str, cache: Path, options: dict):
        import yaml
        from omegaconf import OmegaConf
        from resemble_enhance.enhancer.inference import load_enhancer
        from resemble_enhance.hparams import HParams as BaseHParams

        # Official hyperparameters encode training directories as PosixPath objects.
        # Use a restricted YAML loader and translate those path values portably on Windows.
        class PortableLoader(yaml.SafeLoader):
            pass

        for kind in ["PosixPath", "WindowsPath", "Path"]:
            PortableLoader.add_constructor(
                f"tag:yaml.org,2002:python/object/apply:pathlib.{kind}",
                lambda loader, node: str(Path(*loader.construct_sequence(node))),
            )

        def from_yaml(cls, path):
            with Path(path).open(encoding="utf-8") as stream:
                values = yaml.load(stream, Loader=PortableLoader)
            return cls(**dict(OmegaConf.merge(cls(), values)))

        BaseHParams.from_yaml = classmethod(from_yaml)

        self.device = device
        self.run_dir = cache / "resemble" / "enhancer_stage2"
        from .downloads import download

        for relative in ["hparams.yaml", "ds/G/latest", "ds/G/default/mp_rank_00_model_states.pt"]:
            download(
                f"https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/{relative}?download=true",
                self.run_dir / relative,
            )
        self.model = load_enhancer(self.run_dir, device)
        from .downloads import fingerprint

        self.runtime = {
            "weights": [fingerprint(self.run_dir / "ds/G/default/mp_rank_00_model_states.pt")],
            "hyperparameters": fingerprint(self.run_dir / "hparams.yaml"),
            "windows_path_adapter": True,
        }

    def process(self, source: Path, target: Path, options: dict) -> None:
        import soundfile as sf
        import torch
        from resemble_enhance.enhancer.inference import denoise, enhance

        data, sr = sf.read(source, dtype="float32")
        tensor = torch.from_numpy(data)
        if tensor.ndim != 1:
            raise ValueError("Resemble expects a single channel chunk.")
        if options["denoise"]:
            tensor, sr = denoise(tensor, sr, self.device, run_dir=self.run_dir)
        result, rate = enhance(
            tensor,
            sr,
            self.device,
            nfe=int(options["nfe"]),
            solver=options["solver"],
            lambd=float(options["lambd"]),
            tau=float(options["tau"]),
            run_dir=self.run_dir,
        )
        sf.write(target, result.detach().cpu().numpy(), rate, subtype="FLOAT")
