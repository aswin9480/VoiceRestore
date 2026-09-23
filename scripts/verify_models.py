"""Opt-in integration check using actual upstream models (downloads require --download)."""

import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, Settings, load_config, prepare
from src.job_manager import JobManager
from src.logger import configure_logging
from src.models.model_manager import ModelManager
from src.process import run


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--engine", choices=["clearvoice", "resemble", "both"], default="both")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--longer", action="store_true", help="Exercise real overlapping multi-chunk inference")
    parser.add_argument("--dereverb-model", default="", help="Also test an installed UVR model with the No Reverb stem")
    args = parser.parse_args()
    prepare()
    configure_logging()
    models = ModelManager()
    options = Settings(device=args.device, chunk_seconds=5, overlap_seconds=0.5, nfe=args.nfe)
    options.dereverb = bool(args.dereverb_model)
    options.dereverb_model = args.dereverb_model
    engines = ["clearvoice", "resemble"] if args.engine == "both" else [args.engine]
    try:
        if args.download:
            for engine in engines:
                worker = models.get(engine, args.device, downloads=True)
                print(
                    worker.call(
                        {"action": "load", "options": options.to_dict()},
                        threading.Event(),
                        lambda x: print(x, flush=True),
                    ),
                    flush=True,
                )
                models.unload()
        source = ROOT / "temp" / "integration_speech.wav"
        config = load_config()
        text = "This is a local speech restoration test. Clear speech should sound natural."
        if args.longer:
            text = (
                text
                + " We also test continuity between overlapping chunks and preserve the original recording throughout the entire process."
            )
        # FFmpeg's packaged flite voice generates synthetic speech locally; no user's media leaves the machine.
        run(
            [
                config["ffmpeg"],
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"flite=text={text}:voice=slt",
                "-ar",
                "48000",
                str(source),
            ]
        )
        options.clearvoice = "clearvoice" in engines
        options.resemble = "resemble" in engines
        job = JobManager(config, models).start(str(source), options)
        last = ""
        while not job.done.wait(1):
            text = job.log_text()
            if text != last:
                print(text[len(last) :], flush=True)
                last = text
        print(json.dumps(job.report, indent=2), flush=True)
        if job.report["status"] != "complete":
            raise RuntimeError(job.error)
        print("[OK] Actual model inference and export completed:", job.output)
    finally:
        models.unload()


if __name__ == "__main__":
    main()
