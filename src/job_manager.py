from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT


@dataclass
class Job:
    source: Path
    settings: object
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    lines: deque = field(default_factory=lambda: deque(maxlen=800))
    lock: threading.Lock = field(default_factory=threading.Lock)
    status: str = "Waiting"
    previews: dict = field(default_factory=dict)
    files: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
    output: Path | None = None
    error: str = ""

    def __post_init__(self):
        self.logger = logging.getLogger(f"voicerestore.job.{self.id}")
        self.logger.setLevel(logging.INFO)
        self.logfile = ROOT / "logs" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{self.id}.log"
        self.handler = logging.FileHandler(self.logfile, encoding="utf-8")
        self.handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        self.logger.addHandler(self.handler)

    def log(self, message: str):
        if not message:
            return
        self.logger.info(message)
        with self.lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def log_text(self) -> str:
        with self.lock:
            return "\n".join(self.lines)


class JobManager:
    """One active compute job across browser sessions; STOP can only cancel its owner's job."""

    def __init__(self, config, models):
        self.config, self.models = config, models
        self.lock = threading.RLock()
        self.active: Job | None = None
        self.jobs: dict[str, Job] = {}
        self.maintenance = False

    def start(self, source: str, settings) -> Job:
        from .processing.pipeline import process

        settings.validate()
        path = Path(source).resolve()
        from .config import EXTENSIONS

        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            raise ValueError("Select a supported audio or video file.")
        with self.lock:
            if self.maintenance or (self.active and not self.active.done.is_set()):
                raise ValueError("Another job or model download is active. Wait or stop your current job.")
            job = Job(path, settings)
            self.active = job
            self.jobs[job.id] = job

            def work():
                try:
                    process(job, self.config, self.models)
                except Exception as error:
                    job.status = "Failed: job infrastructure"
                    job.error = f"Could not create or finalize the job: {error}. Check available disk space and folder permissions."
                    job.log(job.error)
                    job.logger.exception("Uncaught job infrastructure failure")
                finally:
                    job.handler.flush()
                    job.logger.removeHandler(job.handler)
                    job.handler.close()
                    job.done.set()

            threading.Thread(target=work, name=f"restore-{job.id}", daemon=True).start()
            return job

    def stop(self, job_id: str | None) -> str:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job.done.is_set():
                return "No active job in this session."
            job.log("Cancellation requested... Stopping active process or processing segment...")
            job.cancel.set()
            return "Cancellation requested."

    def clean(self) -> str:
        with self.lock:
            if self.maintenance or (self.active and not self.active.done.is_set()):
                return "Temporary files cannot be cleaned while a job is active."
            count = 0
            root = (ROOT / "temp").resolve()
            for path in root.iterdir():
                # Only application-owned UUID directories; never traverse reparse points/symlinks.
                try:
                    uuid.UUID(path.name)
                except ValueError:
                    continue
                if path.is_dir() and not path.is_symlink() and path.resolve().parent == root:
                    shutil.rmtree(path)
                    count += 1
            return f"Removed {count} job temporary folders. Browser preview caches are cleared on restart."

    def close(self):
        with self.lock:
            if self.active and not self.active.done.is_set():
                self.active.cancel.set()
                self.active.done.wait(timeout=10)
            self.models.unload()
