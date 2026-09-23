import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.models.model_manager import interpreter
from src.process import run


def main() -> int:
    failed = False
    for name in ["gradio", "numpy", "scipy", "soundfile"]:
        try:
            importlib.import_module(name)
            print(f"[OK] {name}")
        except Exception as error:
            failed = True
            print(f"[ERROR] {name}: {error}. Run Setup / Repair.")
    for name in ["ffmpeg", "ffprobe"]:
        try:
            run([load_config()[name], "-version"], timeout=15)
            print(f"[OK] {name}")
        except Exception as error:
            failed = True
            print(f"[ERROR] {name}: {error}")
    for engine in ["clearvoice", "resemble", "uvr"]:
        print(
            f"[{'OK' if interpreter(engine).exists() else 'WARN'}] {engine} environment "
            + ("installed" if interpreter(engine).exists() else "missing; disable its stage or run Setup")
        )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
