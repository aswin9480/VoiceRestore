"""Exercise the running Gradio HTTP upload, callbacks, session state and result downloads."""

import sys
import time
from pathlib import Path

from gradio_client import Client, handle_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config
from src.process import run


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = Client("http://127.0.0.1:7860", verbose=False)
    source = ROOT / "temp/ui_smoke.wav"
    run(
        [
            load_config()["ffmpeg"],
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3:sample_rate=48000",
            str(source),
        ]
    )
    metadata = client.predict(handle_file(str(source)), api_name="/inspect")
    assert metadata["sample_rate"] == 48000
    params = client.view_api(return_format="dict", print_info=False)["named_endpoints"]["/start"]["parameters"]
    arguments = [handle_file(str(source))]
    for param in params[1:]:
        value = param["parameter_default"]
        if param["label"] in ["ClearVoice Speech Enhancement", "AI Voice Restoration — Resemble Enhance"]:
            value = False
        arguments.append(value)
    print(client.predict(*arguments, api_name="/start"))
    for _ in range(40):
        result = client.predict(api_name="/poll")
        if "Processing Complete" in result[0]:
            assert result[2] and result[3], "Original/final players missing"
            assert len(result[7]) >= 4, "Download files missing"
            print("[OK] UI upload, processing, A/B previews, reports and downloads verified.")
            break
        if "Failed" in result[0]:
            raise RuntimeError(result[0])
        time.sleep(0.5)
    else:
        raise TimeoutError("UI job did not finish")
    client.predict(api_name="/reset")
    print("[OK] UI reset verified.")
    # Exercise STOP through the same session while an actual model worker is starting.
    for index, param in enumerate(params[1:], 1):
        if param["label"] == "ClearVoice Speech Enhancement":
            arguments[index] = True
    client.predict(*arguments, api_name="/start")
    assert "Cancellation requested" in client.predict(api_name="/stop")
    for _ in range(30):
        result = client.predict(api_name="/poll")
        if "Job cancelled" in result[0]:
            print("[OK] UI STOP cancelled the actual job.")
            break
        time.sleep(0.25)
    else:
        raise TimeoutError("STOP failed to cancel the job")
    client.predict(api_name="/reset")


if __name__ == "__main__":
    main()
