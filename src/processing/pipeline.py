from __future__ import annotations

import hashlib
import math
import re
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..config import ROOT, resolve
from ..media.ffmpeg import extract
from ..media.probe import probe
from ..media.remux import remux
from ..process import Cancelled, check_cancel, run
from ..reports.report import write_report
from .chunking import process_chunks
from .mastering import master


def model_stage(source, target, engine, options, job, models, work) -> dict:
    s = job.settings
    if s.low_vram:
        models.unload()
    worker = models.get(engine, s.device)
    metadata = worker.call({"action": "load", "options": options}, job.cancel, job.log)
    if metadata["device"]["fallback"]:
        job.log("Requested CUDA is unavailable or unsupported; using CPU.")
    chunk_in, chunk_out = work / "model_input.wav", work / "model_output.wav"

    def transform(data, sr):
        channels = []
        for channel in range(data.shape[1]):
            check_cancel(job.cancel)
            sf.write(chunk_in, data[:, channel], sr, subtype="FLOAT")
            chunk_out.unlink(missing_ok=True)
            worker.call(
                {"action": "process", "source": str(chunk_in), "target": str(chunk_out), "options": options},
                job.cancel,
                job.log,
            )
            result, rate = sf.read(chunk_out, dtype="float32", always_2d=True)
            result = result.mean(axis=1)
            if rate != sr:
                divisor = math.gcd(rate, sr)
                result = resample_poly(result, sr // divisor, rate // divisor)
            # Resampling / model padding may differ by a few samples, but never accept a truncated chunk.
            if abs(len(result) - len(data)) > max(480, int(len(data) * 0.01)):
                raise RuntimeError(f"Model changed chunk duration ({len(data)} → {len(result)} samples).")
            result = np.pad(result, (0, max(0, len(data) - len(result))))[: len(data)]
            if engine == "uvr" and options.get("kind") == "dereverb":
                wet = {"Conservative": 0.55, "Balanced": 0.75, "Strong": 1.0}[s.aggression]
                result = wet * result + (1 - wet) * data[:, channel]
            channels.append(result)
        return np.stack(channels, axis=1)

    try:
        process_chunks(source, target, transform, s.chunk_seconds, s.overlap_seconds, job.cancel, job.log)
    finally:
        chunk_in.unlink(missing_ok=True)
        chunk_out.unlink(missing_ok=True)
        if s.low_vram:
            models.unload()
    return metadata


def process(job, config, models) -> None:
    started = time.monotonic()
    s = job.settings
    work = ROOT / "temp" / job.id
    work.mkdir(parents=True, exist_ok=False)
    name = re.sub(r"[^\w.-]+", "_", job.source.stem)[:60]
    output = resolve(config["output_directory"]) / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}_{job.id[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    job.output = output
    report = {
        "schema_version": 1,
        "job_id": job.id,
        "status": "running",
        "input": {"filename": job.source.name, "duration": 0},
        "settings": s.to_dict(),
        "pipeline": [],
        "outputs": [],
        "measurements": {},
    }
    stage = "Input validation"
    try:
        job.log(f"Job started. Input: {job.source.name}")
        info = probe(job.source, config, job.cancel)
        report["input"] = info
        current = work / "01_extracted.wav"
        stage = "FFmpeg preprocessing"
        job.status = stage
        job.log(stage)
        extract(job.source, current, info, s.channel, config, job.cancel, job.log)
        job.previews["original"] = str(current)
        report["pipeline"].append({"stage": stage, "status": "complete"})
        stages = [
            ("UVR Vocal Isolation", s.vocal, "uvr", "vocal", s.vocal_model, s.vocal_stem),
            ("ClearVoice", s.clearvoice, "clearvoice", "clearvoice", "MossFormer2_SE_48K", ""),
            ("UVR De-Reverb", s.dereverb, "uvr", "dereverb", s.dereverb_model, s.dereverb_stem),
            ("Resemble Enhance", s.resemble, "resemble", "resemble", "enhancer_stage2", ""),
        ]
        for index, (stage, enabled, engine, key, model, stem) in enumerate(stages, 2):
            entry = {"stage": stage, "enabled": enabled, "status": "disabled"}
            report["pipeline"].append(entry)
            if not enabled:
                job.log(f"{stage} skipped (disabled).")
                continue
            check_cancel(job.cancel)
            job.status = stage
            entry.update(status="running", model=model)
            options = dict(s.to_dict(), model=model, stem=stem, kind=key)
            entry["settings"] = options
            if engine == "uvr":
                from ..models.uvr import model_path as resolve_model

                model_path = resolve_model(s.uvr_directory, model)
                digest = hashlib.sha256()
                with model_path.open("rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        check_cancel(job.cancel)
                        digest.update(block)
                entry["model_sha256"] = digest.hexdigest()
            job.log(f"{stage} started — {model}")
            target = work / f"{index:02d}_{key}.wav"
            metadata = model_stage(current, target, engine, options, job, models, work)
            entry.update(metadata, status="complete")
            current = target
            job.previews[key] = str(current)
            job.log(f"{stage} complete.")
        stage = "Final DSP and loudness"
        job.status = stage
        job.log(stage)
        final = output / "final_restored.wav"
        report["measurements"] = master(current, final, s, config, job.cancel, job.log)
        report["pipeline"].append(
            {
                "stage": stage,
                "status": "complete",
                "settings": {
                    k: getattr(s, k)
                    for k in [
                        "highpass",
                        "highpass_hz",
                        "eq",
                        "eq_amount",
                        "compressor",
                        "limiter",
                        "loudness",
                        "target_lufs",
                    ]
                },
            }
        )
        job.previews["final"] = str(final)
        stage = "Export"
        job.status = stage
        if info["video"]:
            job.log("Remuxing cleaned audio; copying original video stream.")
            export = remux(job.source, final, output, info, config, job.cancel, job.log)
        else:
            export = output / "final_restored.flac"
            run(
                [config["ffmpeg"], "-nostdin", "-v", "error", "-y", "-i", str(final), "-c:a", "flac", str(export)],
                job.cancel,
            )
        check_cancel(job.cancel)
        report["pipeline"].append({"stage": stage, "status": "complete", "video_reencoded": False})
        report["outputs"] = [final.name, export.name, "processing_report.json", "processing_report.txt"]
        if s.save_intermediates:
            destination = output / "intermediate"
            destination.mkdir()
            for key, path in list(job.previews.items()):
                if key != "final":
                    target = destination / Path(path).name
                    shutil.copy2(path, target)
                    job.previews[key] = str(target)
        else:
            # Keep only bounded 60-second previews until reset/cleanup; never retain full intermediates.
            preview_dir = work / "previews"
            preview_dir.mkdir()
            for key, path in list(job.previews.items()):
                if key != "final":
                    preview = preview_dir / f"{key}.wav"
                    run(
                        [
                            config["ffmpeg"],
                            "-nostdin",
                            "-v",
                            "error",
                            "-y",
                            "-i",
                            path,
                            "-t",
                            "60",
                            "-c:a",
                            "pcm_s16le",
                            str(preview),
                        ],
                        job.cancel,
                    )
                    job.previews[key] = str(preview)
        for path in work.glob("*.wav"):
            path.unlink()
        report["status"] = "complete"
        job.status = "Processing Complete"
        job.log("Processing complete. Source file preserved.")
    except Cancelled:
        report["status"] = "cancelled"
        job.status = "Job cancelled"
        job.log("Job cancelled.")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"Processing failed during {stage}. Reason: {error}"
        job.error = (
            report["error"]
            + "\nTry Low VRAM Mode, shorter chunks, CPU mode, or disable the failed stage. For missing dependencies, run Setup / Repair."
        )
        job.status = f"Failed: {stage}"
        job.log(job.error)
        job.logger.error(traceback.format_exc())
        models.unload()
    finally:
        if report["status"] != "complete":
            for entry in report["pipeline"]:
                if entry["status"] == "running":
                    entry["status"] = report["status"]
            job.previews = {}
            # Partial exports must not masquerade as final results.
            for path in output.glob("final_restored.*"):
                path.unlink(missing_ok=True)
            shutil.rmtree(work, ignore_errors=True)
        elif s.save_intermediates:
            shutil.rmtree(work, ignore_errors=True)
        report["processing_seconds"] = time.monotonic() - started
        report["log"] = str(job.logfile)
        write_report(output, report)
        job.report = report
        job.files = (
            [str(output / name) for name in report["outputs"]]
            if report["status"] == "complete"
            else [str(output / "processing_report.json"), str(output / "processing_report.txt")]
        )
        job.files.append(str(job.logfile))
