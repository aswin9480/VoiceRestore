from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_resemble import build  # noqa: E402
from src.config import load_config, prepare  # noqa: E402


def execute(args, log, check=True):
    print("[INFO]", " ".join(str(a) for a in args), flush=True)
    process = subprocess.Popen(
        [str(a) for a in args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    code = process.wait()
    if check and code:
        raise RuntimeError(f"Command failed with exit code {code}: {args[0]}")
    return code


def python_in(name: str) -> Path:
    return ROOT / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--engine", choices=["clearvoice", "resemble", "uvr"])
    args = parser.parse_args()
    prepare()
    if sys.version_info[:2] not in [(3, 10), (3, 11)]:
        print("[ERROR] Install 64-bit Python 3.10 or 3.11, then run Setup again.")
        return 1
    config = load_config()
    failures = []
    with (ROOT / "logs/setup.log").open("a", encoding="utf-8") as log:
        log.write(f"\nSetup started {time.ctime()}\nPython {sys.version}\n")
        print(f"[OK] Python {sys.version.split()[0]}")
        for name in ["ffmpeg", "ffprobe"]:
            try:
                execute([config[name], "-version"], log)
            except (OSError, RuntimeError):
                failures.append(name)
                print(
                    f"[ERROR] {name} missing. Install FFmpeg: winget install Gyan.FFmpeg, reopen launcher, or set config.json paths."
                )
        gpu = bool(shutil.which("nvidia-smi")) and not args.cpu
        if gpu:
            gpu = (
                execute(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], log, check=False)
                == 0
            )
        print(
            "[INFO] Installing CUDA 12.1 wheels; runtime validates GPU and falls back to CPU."
            if gpu
            else "[INFO] Installing CPU wheels."
        )
        envs = [(".venv", None)] if not args.engine else []
        if not args.core_only:
            envs += [
                (f".venv_{engine}", engine)
                for engine in ([args.engine] if args.engine else ["clearvoice", "resemble", "uvr"])
            ]
        for name, engine in envs:
            try:
                python = python_in(name)
                if not python.exists():
                    venv.EnvBuilder(with_pip=True).create(ROOT / name)
                execute([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], log)
                if engine:
                    index = "https://download.pytorch.org/whl/" + ("cu121" if gpu else "cpu")
                    execute(
                        [
                            python,
                            "-m",
                            "pip",
                            "install",
                            "torch==2.5.1",
                            "torchaudio==2.5.1",
                            "torchvision==0.20.1",
                            "--index-url",
                            index,
                        ],
                        log,
                    )
                    if engine == "clearvoice":
                        packages = ["clearvoice==0.1.2", "numpy==1.26.4", "huggingface-hub==0.36.0"]
                    elif engine == "resemble":
                        packages = [str(build(ROOT / "models/cache/wheels"))]
                    else:
                        # 0.30 uses NumPy 1.x and has Windows wheels for its supported dependencies.
                        extra = "gpu" if gpu else "cpu"
                        incompatible = "onnxruntime" if gpu else "onnxruntime-gpu"
                        execute([python, "-m", "pip", "uninstall", "-y", incompatible], log)
                        packages = [
                            f"audio-separator[{extra}]==0.30.2",
                            "numpy==1.26.4",
                            "onnxruntime-gpu==1.20.2" if gpu else "onnxruntime==1.20.1",
                        ]
                    execute([python, "-m", "pip", "install", *packages], log)
                else:
                    execute([python, "-m", "pip", "install", "-r", ROOT / "requirements.txt"], log)
                execute([python, "-m", "pip", "check"], log)
                module = {
                    None: "gradio",
                    "clearvoice": "clearvoice",
                    "resemble": "resemble_enhance.enhancer.inference",
                    "uvr": "audio_separator.separator",
                }[engine]
                execute([python, "-c", f'import {module}; print("[OK] {module} import")'], log)
                freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
                (ROOT / "logs" / f"{name.lstrip('.')}-resolved.txt").write_text(freeze, encoding="utf-8")
                print(f"[OK] {name} ready")
            except Exception as error:
                failures.append(name)
                message = f"[ERROR] {name}: {error}. See logs/setup.log; repair with the same launcher after resolving the reported pip error."
                print(message)
                log.write(message + "\n")
        summary = (
            "[OK] VoiceRestore environment ready. Download models from the UI before the first AI job."
            if not failures
            else "[ERROR] Setup incomplete: " + ", ".join(failures)
        )
        print(summary)
        log.write(summary + "\n")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
