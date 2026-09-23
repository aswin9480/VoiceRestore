from pathlib import Path


def model_path(directory: str, name: str) -> Path:
    from ..config import ROOT, resolve

    prefix = "@studio/"
    root = ROOT / "models/uvr" if name.startswith(prefix) else resolve(directory)
    relative = name[len(prefix) :] if name.startswith(prefix) else name
    target = (root / relative).resolve()
    if (
        not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not target.is_relative_to(root.resolve())
        or not target.is_file()
    ):
        raise ValueError("Select an installed UVR model in Advanced Settings.")
    return target


def scan(directory: str) -> tuple[list[str], list[str]]:
    from ..config import ROOT, resolve

    root = resolve(directory)
    roots = [(root, "")]
    studio = ROOT / "models/uvr"
    if root != studio.resolve():
        roots.append((studio, "@studio/"))
    files = []
    for folder, prefix in roots:
        if folder.is_dir():
            files.extend(
                prefix + p.relative_to(folder).as_posix()
                for p in folder.rglob("*")
                if p.is_file()
                and p.resolve().is_relative_to(folder.resolve())
                and p.suffix.lower() in {".onnx", ".pth", ".ckpt"}
                and "denoise" not in p.name.lower()
            )
    files.sort()
    # Filename heuristics only categorize; upstream validates architecture/checksum at load time.
    dereverb = [
        f
        for f in files
        if any(
            token in f.lower().replace("-", "").replace("_", "") for token in ("dereverb", "deecho", "reverb", "echo")
        )
    ]
    return dereverb, [f for f in files if f not in dereverb]


def stage_local_model(source: Path, cache: Path) -> Path:
    """Reuse installed UVR5 weights/metadata in a private writable cache; never modify UVR5."""
    import hashlib
    import json
    import shutil

    from ..config import ROOT

    key = hashlib.sha256(str(source.parent).encode()).hexdigest()[:16]
    destination = cache / "uvr" / key
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    stamp = target.with_name(target.name + ".source.json")
    identity = {"path": str(source), "size": source.stat().st_size, "modified_ns": source.stat().st_mtime_ns}
    if not target.is_file() or not stamp.exists() or json.loads(stamp.read_text(encoding="utf-8")) != identity:
        print(f"Preparing local model cache: {source.name} (copying existing weights; no download)", flush=True)
        pending = target.with_name(target.name + ".part")
        shutil.copyfile(source, pending)
        pending.replace(target)
        stamp.write_text(json.dumps(identity), encoding="utf-8")
    # Shared downloaded catalogs provide a fallback. Prefer the selected UVR5 model's metadata.
    for directory in [ROOT / "models/uvr", source.parent]:
        for name in ["download_checks.json", "vr_model_data.json", "mdx_model_data.json"]:
            path = directory / name
            if path.is_file() and path.resolve() != (destination / name).resolve():
                shutil.copyfile(path, destination / name)
    metadata = source.parent / "model_data/model_data.json"
    if metadata.exists():
        mapped = "mdx_model_data.json" if source.suffix.lower() in {".onnx", ".ckpt"} else "vr_model_data.json"
        shutil.copyfile(metadata, destination / mapped)
    for folder in [source.parent, source.parent / "model_data/mdx_c_configs"]:
        for path in folder.glob("*.yaml"):
            shutil.copyfile(path, destination / path.name)
    return destination


class UVRAdapter:
    def __init__(self, device: str, cache: Path, options: dict):
        from audio_separator.separator import Separator

        from .downloads import download

        source = model_path(options["uvr_directory"], options["model"])
        filename = source.name
        directory = stage_local_model(source, cache)
        self.separator = Separator(
            model_file_dir=str(directory),
            output_dir=str(cache / "uvr_output"),
            output_format="WAV",
            use_soundfile=True,
            output_single_stem=options["stem"],
            sample_rate=44100,
            normalization_threshold=1.0,
        )
        # Upstream catalogs/configs are cached alongside local weights; no metadata network dependency after verification.
        self.separator.download_file_if_not_exists = lambda url, output_path: download(url, Path(output_path))
        if not self.separator.onnx_execution_provider:
            self.separator.onnx_execution_provider = ["CPUExecutionProvider"]
        self.separator.load_model(model_filename=filename)
        instance = self.separator.model_instance
        self.runtime = {
            "source_model": str(source),
            "torch_device": str(self.separator.torch_device),
            "onnx_providers": self.separator.onnx_execution_provider,
            "primary_stem": getattr(instance, "primary_stem_name", None),
            "secondary_stem": getattr(instance, "secondary_stem_name", None),
        }
        valid = [self.runtime["primary_stem"], self.runtime["secondary_stem"]]
        if all(valid) and options["stem"].lower() not in [str(stem).lower() for stem in valid]:
            raise ValueError(
                f"The requested stem {options['stem']!r} does not exist in this model. Available stems: {valid}"
            )

    def process(self, source: Path, target: Path, options: dict) -> None:
        import shutil

        import numpy as np
        import soundfile as sf

        self.separator.output_dir = str(target.parent)
        if self.separator.model_instance is not None:
            self.separator.model_instance.output_dir = str(target.parent)

        # 0.30.2's soundfile writer ignores output_dir and can flatten/cast stereo float
        # data to int16. Preserve the actual model output and propagate export errors.
        def write_audio(stem_path, samples):
            path = Path(stem_path)
            if path.resolve() != target.resolve():
                raise RuntimeError(f"Unexpected UVR output path: {path}")
            samples = np.asarray(samples, dtype=np.float32)
            if samples.ndim not in (1, 2) or not np.isfinite(samples).all():
                raise ValueError("UVR produced invalid audio samples.")
            sf.write(path, samples, self.separator.sample_rate, subtype="FLOAT")

        self.separator.model_instance.write_audio = write_audio
        paths = self.separator.separate(str(source), custom_output_names={options["stem"]: str(target.with_suffix(""))})
        resolved = [Path(p) if Path(p).is_absolute() else target.parent / p for p in paths]
        if len(resolved) != 1:
            raise RuntimeError(
                f"UVR returned {len(resolved)} stems instead of the requested {options['stem']!r}. "
                "Set the exact dry/vocal stem label from the model documentation in Advanced Settings."
            )
        if not resolved[0].exists():
            raise RuntimeError("UVR did not write the selected stem.")
        if resolved[0].resolve() != target.resolve():
            shutil.move(str(resolved[0]), target)
