from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

import gradio as gr

from ..config import EXTENSIONS, Settings, save_config
from ..media.probe import probe
from ..models.uvr import scan
from ..processing.presets import PRESETS, RESTORATION, pipeline_text, preset
from .components import CSS, waveform


def build_ui(config, manager, models):
    defaults = preset(config["default_preset"])
    defaults.device = config["device"]
    defaults.low_vram = config["low_vram"]
    defaults.save_intermediates = config["save_intermediates"]
    defaults.uvr_directory = config["uvr_model_directory"]
    controls = {}

    def settings(values):
        return Settings(**dict(zip(controls, values)))

    with gr.Blocks(
        title="VoiceRestore Studio",
        theme=gr.themes.Base(
            primary_hue="teal",
            neutral_hue="slate",
            font=["Segoe UI", "Arial", "sans-serif"],
            font_mono=["Consolas", "monospace"],
        ),
        css=CSS,
        analytics_enabled=False,
    ) as app:
        job_id = gr.State(None)
        displayed = gr.State(None)
        gr.HTML(
            '<div id="hero"><div class="eyebrow">LOCAL AUDIO WORKSPACE</div><h1>VoiceRestore Studio</h1>'
            "<p><strong>AI Speech Isolation &amp; Restoration</strong><br>Restore noisy, muffled and reverberant speech recordings locally.</p></div>"
        )
        with gr.Row():
            with gr.Column(scale=4, elem_classes="pipeline-card"):
                gr.Markdown("### 01 · Your recording")
                upload = gr.File(label="Drop an audio or video file", file_types=EXTENSIONS, type="filepath")
                info = gr.JSON(label="Input information", value={})
                controls["channel"] = gr.Dropdown(
                    ["Auto", "Stereo → Mono", "Left Channel", "Right Channel", "Keep Stereo"],
                    value="Auto",
                    label="Channel Handling",
                    info="Auto checks for dead channels and phase cancellation. Mono is recommended for speech.",
                )
            with gr.Column(scale=4, elem_classes="pipeline-card"):
                gr.Markdown("### 02 · Processing pipeline")
                for key, label in [
                    ("clearvoice", "ClearVoice Speech Enhancement"),
                    ("dereverb", "UVR De-Reverb"),
                    ("vocal", "UVR Vocal Isolation"),
                    ("resemble", "AI Voice Restoration — Resemble Enhance"),
                    ("highpass", "High-Pass Filter"),
                    ("eq", "Speech Clarity EQ"),
                    ("compressor", "Compressor"),
                    ("limiter", "Limiter"),
                    ("loudness", "Loudness Normalization"),
                ]:
                    controls[key] = gr.Checkbox(value=getattr(defaults, key), label=label)
                gr.Markdown(
                    "Vocal isolation is only recommended for strong music or competing non-speech audio. Leave it off for ordinary noisy speech."
                )
            with gr.Column(scale=3, elem_classes="pipeline-card"):
                gr.Markdown("### 03 · The right touch")
                controls["preset"] = gr.Dropdown(PRESETS, value=defaults.preset, label="Preset")
                gr.Markdown(
                    "**Natural speech comes first.** Start conservatively and compare the original before adding more restoration."
                )
                preset_warning = gr.Markdown("")
                device_status = gr.Markdown("**Compute Device**\n\nChecking model environment…")
                pipeline = gr.Textbox(
                    value=pipeline_text(defaults), label="Current Pipeline", lines=12, interactive=False
                )
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Dropdown(
                        ["MossFormer2_SE_48K"], value="MossFormer2_SE_48K", label="ClearVoice Model", interactive=False
                    )
                    controls["device"] = gr.Dropdown(
                        ["auto", "cuda", "cpu"], value=defaults.device, label="Compute preference"
                    )
                    controls["low_vram"] = gr.Checkbox(
                        defaults.low_vram, label="Low VRAM Mode", info="Unload each model before the next stage."
                    )
                    controls["chunk_seconds"] = gr.Slider(5, 120, value=30, step=1, label="Chunk Length (seconds)")
                    controls["overlap_seconds"] = gr.Slider(0, 5, value=1, step=0.1, label="Overlap (seconds)")
                    controls["save_intermediates"] = gr.Checkbox(
                        defaults.save_intermediates, label="Save intermediate files"
                    )
                    gr.Markdown(
                        "Keep Stereo processes each channel independently through speech models; spatial character may change."
                    )
                with gr.Column():
                    controls["restoration"] = gr.Dropdown(
                        list(RESTORATION), value=defaults.restoration, label="Resemble preset"
                    )
                    controls["denoise"] = gr.Checkbox(
                        False,
                        label="Denoise before enhancement",
                        info="Off by default to avoid stacking strong denoisers.",
                    )
                    controls["solver"] = gr.Dropdown(["midpoint", "rk4", "euler"], value="midpoint", label="Solver")
                    controls["nfe"] = gr.Slider(1, 128, value=defaults.nfe, step=1, label="NFE")
                    controls["tau"] = gr.Slider(0, 1, value=defaults.tau, step=0.05, label="Prior Temperature / Tau")
                    controls["lambd"] = gr.Slider(
                        0,
                        1,
                        value=defaults.lambd,
                        step=0.05,
                        label="Lambda — internal denoiser mix",
                        info="Upstream Lambda controls denoiser conditioning, not a general restoration strength.",
                    )
                with gr.Column():
                    controls["highpass_hz"] = gr.Slider(
                        20, 250, value=defaults.highpass_hz, step=1, label="High-pass frequency (Hz)"
                    )
                    controls["eq_amount"] = gr.Slider(
                        0, 1.5, value=defaults.eq_amount, step=0.1, label="Speech EQ amount"
                    )
                    controls["target_lufs"] = gr.Slider(
                        -30, -10, value=defaults.target_lufs, step=0.5, label="Target loudness (LUFS)"
                    )
                    gr.Markdown(
                        "Compression: −18 dB, 2.5:1, 10 ms attack, 120 ms release.\n\nLimiter: −1 dBTP target with 4× oversampling. Actual final loudness and true peak are measured in the report."
                    )
            gr.Markdown("#### Local UVR models")
            controls["uvr_directory"] = gr.Textbox(
                defaults.uvr_directory,
                label="UVR Model Directory",
                info="Paste your UVR5 models folder or another model directory. Subfolders are scanned; Studio de-reverb models remain available under @studio/.",
            )
            refresh = gr.Button("Scan Model Directory")
            dereverb_choices, vocal_choices = scan(defaults.uvr_directory)
            with gr.Row():
                controls["dereverb_model"] = gr.Dropdown(
                    dereverb_choices, value=config.get("last_dereverb_model") or None, label="UVR De-Reverb Model"
                )
                controls["vocal_model"] = gr.Dropdown(
                    vocal_choices, value=config.get("last_vocal_model") or None, label="UVR Vocal Isolation Model"
                )
                controls["aggression"] = gr.Dropdown(
                    ["Conservative", "Balanced", "Strong"], value="Conservative", label="De-Reverb Aggressiveness"
                )
            with gr.Row():
                controls["dereverb_stem"] = gr.Textbox(
                    "No Reverb",
                    label="Dry stem name",
                    info="Exact label required by your model, e.g. No Reverb, No Echo, or Vocals. Consult its model card.",
                )
                controls["vocal_stem"] = gr.Textbox("Vocals", label="Vocal stem name")
            gr.Markdown(
                "Scanned filenames are candidates. The inference engine validates model compatibility when loaded. Only use trusted model weights; PyTorch checkpoints may contain executable objects."
            )
        with gr.Row():
            process_button = gr.Button("PROCESS RECORDING", variant="primary", elem_id="process", scale=4)
            stop_button = gr.Button("STOP", variant="stop", scale=1)
            reset_button = gr.Button("RESET", scale=1)
        status = gr.Markdown("Ready when you are. Select a recording to begin.")
        console = gr.Textbox(
            label="Live Processing Log", lines=10, max_lines=16, interactive=False, elem_id="console", autoscroll=True
        )
        with gr.Tab("Listen & compare"):
            gr.Markdown(
                "### A/B · Original vs Final\nPreview the same opening segment. Full files are available below; intermediate previews are limited to 60 seconds unless you save intermediates."
            )
            with gr.Row():
                with gr.Column():
                    original_player = gr.Audio(label="A · Original Audio", type="filepath", interactive=False)
                    original_wave = gr.HTML()
                with gr.Column():
                    final_player = gr.Audio(label="B · Final Master", type="filepath", interactive=False)
                    final_wave = gr.HTML()
            with gr.Row():
                cv_player = gr.Audio(label="After ClearVoice", type="filepath", interactive=False, visible=False)
                uvr_player = gr.Audio(label="After UVR De-Reverb", type="filepath", interactive=False, visible=False)
                re_player = gr.Audio(label="After Resemble Enhance", type="filepath", interactive=False, visible=False)
            results = gr.Markdown("Your restored recording will appear here.")
            downloads = gr.File(
                label="Download final media, reports and job log", file_count="multiple", interactive=False
            )
            open_folder = gr.Button("Open Output Folder")
        with gr.Tab("Processing report"):
            report_ui = gr.JSON(label="Exact settings, model versions, device and measured output")
        with gr.Accordion("Model Status & Maintenance", open=False):
            model_status = gr.Markdown(models.status())
            gr.Markdown(
                "Downloads contact the official model repositories. Your recordings stay on this computer. Models are cached and reused; first download can take several minutes."
            )
            download_engines = gr.CheckboxGroup(
                ["clearvoice", "resemble", "uvr"],
                value=["clearvoice", "resemble"],
                label="Models to download / verify",
                info="UVR verifies your selected local models and caches their companion configuration files.",
            )
            with gr.Row():
                download_button = gr.Button("Download / Verify Models")
                cancel_download = gr.Button("Cancel Download")
                unload_button = gr.Button("Unload Models")
                clean_button = gr.Button("Clean Temporary Files")
            maintenance = gr.Textbox(label="Model and maintenance log", lines=8, interactive=False, autoscroll=True)
        gr.HTML(
            '<div class="footer">All processing is performed locally on this computer. No accounts. No analytics. No audio uploads to external services.</div>'
        )
        timer = gr.Timer(0.75)

        values = list(controls.values())

        def inspect(path):
            if not path:
                return {}
            try:
                return probe(Path(path), config)
            except Exception as error:
                return {"error": str(error)}

        upload.change(inspect, upload, info, queue=False)

        def show_pipeline(*vals):
            return pipeline_text(settings(vals))

        for control in values:
            control.change(show_pipeline, values, pipeline, queue=False)
        preset_keys = [
            "clearvoice",
            "dereverb",
            "vocal",
            "resemble",
            "highpass",
            "highpass_hz",
            "eq",
            "eq_amount",
            "compressor",
            "limiter",
            "loudness",
            "target_lufs",
            "restoration",
            "nfe",
            "tau",
            "lambd",
            "denoise",
            "solver",
        ]

        def apply_preset(name):
            if name == "Custom":
                return [gr.skip() for _ in preset_keys] + [""]
            selected = preset(name)
            return [getattr(selected, key) for key in preset_keys] + [
                "Strong processing may introduce speech artifacts." if name == "Strong Cleanup" else ""
            ]

        controls["preset"].change(
            apply_preset, controls["preset"], [controls[k] for k in preset_keys] + [preset_warning], queue=False
        )
        controls["restoration"].change(
            lambda name: RESTORATION[name],
            controls["restoration"],
            [controls[k] for k in ["nfe", "tau", "lambd"]],
            queue=False,
        )

        def scan_models(directory):
            a, b = scan(directory)
            return gr.update(choices=a, value=None), gr.update(choices=b, value=None)

        refresh.click(
            scan_models, controls["uvr_directory"], [controls["dereverb_model"], controls["vocal_model"]], queue=False
        )

        def start(path, *vals):
            if not path:
                raise gr.Error("Choose an audio or video file first.")
            try:
                s = settings(vals)
                s.dereverb_model = s.dereverb_model or ""
                s.vocal_model = s.vocal_model or ""
                job = manager.start(path, s)
                config.update(
                    default_preset=s.preset,
                    device=s.device,
                    low_vram=s.low_vram,
                    save_intermediates=s.save_intermediates,
                    uvr_model_directory=s.uvr_directory,
                    last_dereverb_model=s.dereverb_model,
                    last_vocal_model=s.vocal_model,
                )
                save_config(config)
                return job.id, None, "**Processing started…**"
            except Exception as error:
                raise gr.Error(str(error)) from error

        process_button.click(start, [upload] + values, [job_id, displayed, status], queue=False)
        stop_button.click(manager.stop, job_id, status, queue=False)
        result_outputs = [
            status,
            console,
            displayed,
            original_player,
            final_player,
            cv_player,
            uvr_player,
            re_player,
            downloads,
            report_ui,
            results,
            original_wave,
            final_wave,
        ]

        def poll(identifier, shown):
            job = manager.jobs.get(identifier)
            if not job:
                return [gr.skip()] * len(result_outputs)
            lead = [f"**{job.status}**" + ("\n\n" + job.error if job.error else ""), job.log_text()]
            if not job.done.is_set() or shown == identifier:
                return lead + [gr.skip()] * (len(result_outputs) - 2)
            p = job.previews
            measured = job.report.get("measurements", {})
            text = (
                f"**{job.status}** · {job.report.get('processing_seconds', 0):.1f}s processing\n\n"
                f"Original duration: {job.report['input']['duration']:.1f}s · Input: {job.report['input'].get('format', 'unknown')}\n\n"
                f"Measured loudness: **{measured.get('integrated_lufs', 'N/A')} LUFS** · True peak: **{measured.get('true_peak_dbtp', 'N/A')} dBTP**"
            )
            waves = []
            for key, label in [("original", "Original"), ("final", "Final")]:
                try:
                    waves.append(waveform(p[key], label) if key in p else "")
                except Exception:
                    waves.append("Waveform unavailable. Audio files remain available.")
            return lead + [
                identifier,
                p.get("original"),
                p.get("final"),
                *[gr.update(value=p.get(k), visible=k in p) for k in ["clearvoice", "dereverb", "resemble"]],
                job.files,
                job.report,
                text,
                *waves,
            ]

        timer.tick(poll, [job_id, displayed], result_outputs, queue=False, show_progress="hidden")

        def reset(identifier):
            job = manager.jobs.get(identifier)
            if job and not job.done.is_set():
                raise gr.Error("Stop the active job before resetting.")
            return [
                None,
                None,
                None,
                {},
                "Ready when you are.",
                "",
                None,
                None,
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                None,
                {},
                "",
                "",
                "",
            ]

        reset_button.click(
            reset,
            job_id,
            [
                job_id,
                displayed,
                upload,
                info,
                status,
                console,
                original_player,
                final_player,
                cv_player,
                uvr_player,
                re_player,
                downloads,
                report_ui,
                results,
                original_wave,
                final_wave,
            ],
            queue=False,
        )

        def folder(identifier):
            job = manager.jobs.get(identifier)
            if not job or not job.output:
                return "No output folder yet."
            if os.name == "nt":
                os.startfile(job.output)
            return f"Output folder: {job.output}"

        open_folder.click(folder, job_id, status, queue=False)
        clean_button.click(manager.clean, outputs=maintenance, queue=False)

        def unload():
            with manager.lock:
                if manager.maintenance or (manager.active and not manager.active.done.is_set()):
                    return "Wait until the active job or download finishes."
                models.unload()
                return "Models unloaded. GPU memory released."

        unload_button.click(unload, outputs=maintenance, queue=False)
        download_cancel = threading.Event()

        def download(engines, *vals):
            selected = settings(vals)
            with manager.lock:
                if manager.maintenance or (manager.active and not manager.active.done.is_set()):
                    raise gr.Error("Wait until the active job finishes.")
                manager.maintenance = True
            download_cancel.clear()
            messages = queue.Queue()
            finished = threading.Event()

            def work():
                try:
                    models.unload()
                    for engine in engines:
                        worker = models.get(engine, selected.device, downloads=True)
                        options = [selected.to_dict()]
                        if engine == "uvr":
                            options = [
                                dict(selected.to_dict(), model=model, stem=stem)
                                for model, stem in [
                                    (selected.dereverb_model, selected.dereverb_stem),
                                    (selected.vocal_model, selected.vocal_stem),
                                ]
                                if model
                            ]
                            if not options:
                                raise ValueError("Select a local UVR model under Advanced Settings first.")
                        for option in options:
                            worker.call({"action": "load", "options": option}, download_cancel, messages.put)
                        messages.put(f"{engine}: downloaded and loaded successfully.")
                        models.unload()
                except Exception as error:
                    messages.put(f"Download / validation stopped: {error}")
                finally:
                    models.unload()
                    manager.maintenance = False
                    finished.set()

            threading.Thread(target=work, daemon=True).start()
            lines = []
            while not finished.is_set() or not messages.empty():
                try:
                    lines.append(messages.get(timeout=0.25))
                except queue.Empty:
                    pass
                yield "\n".join(lines[-200:]), models.status()
            yield "\n".join(lines[-200:]), models.status()

        download_button.click(download, [download_engines] + values, [maintenance, model_status])
        cancel_download.click(
            lambda: (download_cancel.set(), "Download cancellation requested.")[1], outputs=maintenance, queue=False
        )

        def detect():
            from ..models.model_manager import interpreter
            from ..process import run

            for engine in ["clearvoice", "resemble", "uvr"]:
                if interpreter(engine).exists():
                    try:
                        code = "import json; from src.device import torch_device; print(json.dumps(torch_device()))"
                        data = json.loads(run([str(interpreter(engine)), "-c", code], timeout=45))
                        return f"**Compute Device**\n\n{data['gpu'] or 'CPU Mode'} · {data['type'].upper()}"
                    except Exception:
                        continue
            return "**Compute Device**\n\nCPU Mode · AI environments need Setup / Repair"

        app.load(detect, outputs=device_status)
        app.load(None, js="() => {document.body.classList.add('dark');}")
    return app
