"""Opt-in real UVR integration test; fetch an official catalog model for testing."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config, prepare
from src.device import torch_device
from src.models.uvr import UVRAdapter
from src.process import run


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    prepare()
    os.environ["HF_HUB_OFFLINE"] = "0" if args.download else "1"
    from audio_separator.separator import Separator

    from src.models.downloads import download

    root = ROOT / "models/uvr"
    catalog = Separator(model_file_dir=str(root), info_only=True)
    catalog.download_file_if_not_exists = lambda url, output_path: download(url, Path(output_path))
    models = catalog.list_supported_model_files()
    candidates = [
        (name, model)
        for entries in models.values()
        for name, model in entries.items()
        if "deecho" in name.lower().replace("-", "").replace("_", "")
        and "dereverb" in name.lower().replace("-", "").replace("_", "")
    ]
    if not candidates:
        raise RuntimeError("No matching de-echo/de-reverb test model in the current upstream catalog.")
    name, model = candidates[0]
    print("Verifying catalog model:", name, model["filename"], flush=True)
    catalog.download_model_and_data(model["filename"])
    options = {"uvr_directory": str(root), "model": model["filename"], "stem": "No Reverb"}
    device = torch_device()["type"]
    adapter = UVRAdapter(device, ROOT / "models/cache", options)
    print("Actual model runtime:", json.dumps(adapter.runtime), flush=True)
    source, target = ROOT / "temp/uvr_speech.wav", ROOT / "temp/uvr_verified.wav"
    run(
        [
            load_config()["ffmpeg"],
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "flite=text=This is a local echo removal test.:voice=slt",
            "-ar",
            "48000",
            str(source),
        ]
    )
    adapter.process(source, target, options)
    import numpy as np
    import soundfile as sf

    result, sr = sf.read(target)
    assert len(result) > 1000 and np.isfinite(result).all()
    print("[OK] Actual UVR inference:", target, sr, len(result))


if __name__ == "__main__":
    main()
