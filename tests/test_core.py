import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.config import Settings, load_config, prepare
from src.job_manager import JobManager
from src.logger import configure_logging
from src.media.ffmpeg import extract
from src.media.probe import probe
from src.models.model_manager import ModelManager
from src.process import Cancelled, run
from src.processing.chunking import process_chunks


@pytest.fixture
def config(tmp_path):
    prepare()
    configure_logging()
    return dict(load_config(), output_directory=str(tmp_path / "exports"))


@pytest.fixture
def speech(tmp_path):
    sr = 48000
    t = np.arange(sr * 4) / sr
    # Varying amplitude and harmonics exercise dynamics and normalization.
    audio = (0.15 * np.sin(2 * np.pi * 220 * t) + 0.035 * np.sin(2 * np.pi * 3400 * t)) * (
        0.6 + 0.4 * np.sin(2 * np.pi * 3 * t) ** 2
    )
    path = tmp_path / "recording with spaces.wav"
    sf.write(path, audio, sr, subtype="PCM_24")
    return path


@pytest.mark.parametrize("length,overlap", [(500, 0), (500, 100), (500, 249), (5000, 100), (501, 100)])
def test_chunk_identity_preserves_samples_and_channels(tmp_path, length, overlap):
    rng = np.random.default_rng(42)
    data = rng.uniform(-0.5, 0.5, (1703, 2)).astype("float32")
    source, output = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(source, data, 1000, subtype="FLOAT")
    process_chunks(
        source, output, lambda x, sr: x.copy(), length / 1000, overlap / 1000, threading.Event(), lambda x: None
    )
    actual, _ = sf.read(output, always_2d=True)
    assert actual.shape == data.shape
    np.testing.assert_allclose(actual, data, atol=1e-7)


def test_chunk_rejects_invalid_model_output(tmp_path, speech):
    with pytest.raises(ValueError, match="invalid audio"):
        process_chunks(speech, tmp_path / "bad.wav", lambda x, sr: x[:-20], 1, 0.1, threading.Event(), lambda x: None)


def test_auto_avoids_antiphase_cancellation(tmp_path, config):
    signal = np.sin(np.arange(48000) * 0.05).astype("float32") * 0.2
    source, output = tmp_path / "phase.wav", tmp_path / "mono.wav"
    sf.write(source, np.stack([signal, -signal], axis=1), 48000, subtype="FLOAT")
    extract(source, output, probe(source, config), "Auto", config, threading.Event(), lambda x: None)
    actual, _ = sf.read(output)
    assert np.sqrt(np.mean(actual**2)) > 0.1


def wait_job(job):
    assert job.done.wait(40), "Job did not finish in time"
    assert job.status == "Processing Complete", job.error


def test_real_audio_pipeline(config, speech):
    original_hash = hashlib.sha256(speech.read_bytes()).hexdigest()
    models = ModelManager()
    manager = JobManager(config, models)
    job = manager.start(str(speech), Settings(clearvoice=False, resemble=False))
    wait_job(job)
    assert hashlib.sha256(speech.read_bytes()).hexdigest() == original_hash
    assert sf.info(job.output / "final_restored.wav").subtype == "PCM_24"
    assert sf.info(job.output / "final_restored.wav").samplerate == 48000
    assert (job.output / "final_restored.flac").exists()
    report = json.loads((job.output / "processing_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert abs(float(report["measurements"]["integrated_lufs"]) + 16) < 0.5
    assert float(report["measurements"]["true_peak_dbtp"]) <= -0.8
    assert not (job.output / "intermediate").exists()
    assert all(Path(path).exists() for path in job.previews.values())
    manager.close()


def test_video_stream_is_not_reencoded(tmp_path, config, speech):
    source = tmp_path / "video.mp4"
    run(
        [
            config["ffmpeg"],
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:r=10:d=4",
            "-i",
            str(speech),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ]
    )
    models = ModelManager()
    job = JobManager(config, models).start(
        str(source), Settings(clearvoice=False, resemble=False, save_intermediates=True)
    )
    wait_job(job)
    output = job.output / "final_restored.mp4"
    assert output.exists()

    def bitstream_hash(path):
        return run([config["ffmpeg"], "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "hash", "-"])

    assert bitstream_hash(source) == bitstream_hash(output)
    assert (job.output / "intermediate").exists()
    models.unload()


def test_subprocess_cancellation_is_prompt():
    cancel = threading.Event()
    timer = threading.Timer(0.3, cancel.set)
    timer.start()
    start = time.monotonic()
    with pytest.raises(Cancelled):
        run([sys.executable, "-c", "import time; time.sleep(30)"], cancel)
    assert time.monotonic() - start < 5
    timer.join()


def test_failed_model_does_not_silently_skip(config, speech, monkeypatch):
    models = ModelManager()

    def fail(*args, **kwargs):
        raise RuntimeError("Deliberate unavailable model for failure-path test")

    monkeypatch.setattr(models, "get", fail)
    job = JobManager(config, models).start(str(speech), Settings())
    assert job.done.wait(20)
    assert job.status == "Failed: ClearVoice"
    assert not (job.output / "final_restored.wav").exists()
    assert job.report["status"] == "failed"
    assert "Deliberate unavailable" in job.report["error"]
    assert job.report["pipeline"][-1]["status"] == "failed"


def test_silence_is_preserved(tmp_path, config):
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros(48000), 48000)
    models = ModelManager()
    job = JobManager(config, models).start(str(source), Settings(clearvoice=False, resemble=False))
    wait_job(job)
    assert job.report["measurements"]["normalization_applied"] is False
    assert np.max(np.abs(sf.read(job.output / "final_restored.wav")[0])) == 0


def test_invalid_uvr_path_rejected():
    with pytest.raises(ValueError, match="installed UVR"):
        Settings(dereverb=True, dereverb_model="../outside.pth").validate()
