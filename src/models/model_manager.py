from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

from ..config import ROOT
from ..process import check_cancel


def interpreter(engine: str) -> Path:
    root = ROOT / f".venv_{engine}"
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class Worker:
    def __init__(self, engine: str, device: str, downloads: bool):
        python = interpreter(engine)
        if not python.exists():
            raise RuntimeError(f"{engine} environment is not installed. Run Setup / Repair from VoiceRestore.bat.")
        self.lock = threading.Lock()
        self.responses = queue.Queue()
        self.log = lambda line: None
        self.process = subprocess.Popen(
            [str(python), "-u", "-m", "src.models.worker", engine, device, str(downloads).lower()],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        def read_responses():
            for line in self.process.stdout:
                try:
                    self.responses.put(json.loads(line))
                except json.JSONDecodeError:
                    self.log(line.rstrip())
            self.responses.put(
                {"ok": False, "error": "Model worker exited. Check the job log for dependency or GPU errors."}
            )

        def read_logs():
            for line in self.process.stderr:
                self.log(line.rstrip())

        self.threads = [
            threading.Thread(target=read_responses, daemon=True),
            threading.Thread(target=read_logs, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def call(self, request: dict, cancel: threading.Event, log) -> dict:
        with self.lock:
            self.log = log
            check_cancel(cancel)
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                while True:
                    check_cancel(cancel)
                    try:
                        response = self.responses.get(timeout=0.1)
                        break
                    except queue.Empty:
                        continue
                if not response["ok"]:
                    raise RuntimeError(response["error"])
                return response["result"]
            except BaseException:
                self.close()
                raise
            finally:
                self.log = lambda line: None

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for thread in self.threads:
            thread.join(timeout=2)
        for stream in [self.process.stdin, self.process.stdout, self.process.stderr]:
            if stream:
                stream.close()


class ModelManager:
    def __init__(self):
        self.workers: dict[tuple, Worker] = {}
        self.lock = threading.RLock()
        atexit.register(self.unload)

    def get(self, engine, device, downloads=False) -> Worker:
        key = (engine, device, downloads)
        with self.lock:
            worker = self.workers.get(key)
            if worker is None or worker.process.poll() is not None:
                worker = self.workers[key] = Worker(engine, device, downloads)
            return worker

    def unload(self) -> None:
        with self.lock:
            for worker in self.workers.values():
                worker.close()
            self.workers.clear()

    def status(self) -> str:
        statuses = []
        for engine in ["clearvoice", "resemble", "uvr"]:
            installed = interpreter(engine).exists()
            required = {
                "clearvoice": "clearvoice/MossFormer2_SE_48K/last_best_checkpoint",
                "resemble": "resemble/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt",
            }
            ready = (
                ROOT.joinpath(f"models/cache/{engine}.ready.json").exists()
                and engine in required
                and ROOT.joinpath("models/cache", required[engine]).is_file()
            )
            status = "Cached (previously loaded successfully)" if ready else "Not downloaded / not verified"
            if engine == "uvr":
                status = "Local models — select and validate at load"
            statuses.append(
                f"**{engine.title()}**: {status}" if installed else f"**{engine.title()}**: environment not installed"
            )
        return "\n\n".join(statuses)
