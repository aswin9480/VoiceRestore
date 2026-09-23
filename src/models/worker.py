"""Persistent JSON-lines model RPC; stdout is protocol-only, diagnostics go to stderr."""

from __future__ import annotations

import contextlib
import gc
import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> None:
    engine, preference, allow_download = sys.argv[1:4]
    from ..config import ROOT

    cache = ROOT / "models/cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "0" if allow_download == "true" else "1"
    if preference == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Legacy upstream checkpoints contain trusted upstream Python metadata.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from ..device import torch_device

        device = torch_device(preference)
        import torch

        if device["type"] == "cpu":
            torch.cuda.is_available = lambda: False
        adapters = {}
        for line in sys.stdin:
            try:
                request = json.loads(line)
                action = request["action"]
                if action == "info":
                    result = device
                elif action == "unload":
                    adapters.clear()
                    if engine == "resemble":
                        from resemble_enhance.denoiser.inference import load_denoiser
                        from resemble_enhance.enhancer.inference import load_enhancer

                        load_enhancer.cache_clear()
                        load_denoiser.cache_clear()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    result = {"unloaded": True}
                else:
                    options = request["options"]
                    key = json.dumps({k: options.get(k) for k in ["model", "stem", "uvr_directory"]}, sort_keys=True)
                    if key not in adapters:
                        print(f"Loading {engine} on {device['type']}...", flush=True)
                        if engine == "clearvoice":
                            from .clearvoice import ClearVoiceAdapter as Adapter
                        elif engine == "resemble":
                            from .resemble import ResembleAdapter as Adapter
                        else:
                            from .uvr import UVRAdapter as Adapter
                        adapters[key] = Adapter(device["type"], cache, options)
                        print("Model loaded.", flush=True)
                        if engine != "uvr":
                            (cache / f"{engine}.ready.json").write_text(
                                json.dumps({"engine": engine, "device": device}), encoding="utf-8"
                            )
                    package = {"clearvoice": "clearvoice", "resemble": "resemble-enhance", "uvr": "audio-separator"}[
                        engine
                    ]
                    result = {"device": device, "package": package, "version": importlib.metadata.version(package)}
                    if hasattr(adapters[key], "runtime"):
                        result["runtime"] = adapters[key].runtime
                    if action == "process":
                        target = Path(request["target"])
                        adapters[key].process(Path(request["source"]), target, options)
                        if not target.is_file():
                            raise RuntimeError("Model returned without creating output audio.")
                response = {"ok": True, "result": result}
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            protocol.write(json.dumps(response) + "\n")
            protocol.flush()


if __name__ == "__main__":
    main()
