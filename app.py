from __future__ import annotations

import argparse
import atexit
import socket

from src.config import ROOT, load_config, prepare, resolve


def available_port(preferred: int) -> int:
    for port in range(preferred, min(preferred + 100, 65536)):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available localhost port. Change port in config.json.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    prepare()
    from src.job_manager import JobManager
    from src.logger import configure_logging
    from src.models.model_manager import ModelManager
    from src.ui.gradio_app import build_ui

    configure_logging()
    config = load_config()
    models = ModelManager()
    manager = JobManager(config, models)
    atexit.register(manager.close)
    app = build_ui(config, manager, models)
    app.queue(default_concurrency_limit=1, max_size=8)
    port = available_port(int(config["port"]))
    print(f"[OK] VoiceRestore Studio: http://127.0.0.1:{port}")
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            inbrowser=not args.no_browser,
            share=False,
            allowed_paths=[str(resolve(config["output_directory"])), str(ROOT / "temp"), str(ROOT / "logs")],
            blocked_paths=[str(ROOT / ".venv"), str(ROOT / "models"), str(ROOT / "config.local.json")],
            show_error=False,
            quiet=True,
        )
    finally:
        manager.close()


if __name__ == "__main__":
    main()
