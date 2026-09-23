# VoiceRestore Studio

A local Windows speech-restoration workspace for phone recordings, lectures, meetings and reverberant halls. A Gradio interface controls FFmpeg, ClearVoice MossFormer2_SE_48K, optional UVR de-reverberation, and optional Resemble Enhance. The source recording is never modified.

## System requirements

The software requirements below are enforced by this project's setup. CPU, RAM and free-space figures are **practical planning guidelines**, not benchmark-certified minimums. Memory use varies with the selected model, chunk length, stereo processing and other running applications.

| Component | Practical starting configuration | Recommended for regular AI restoration |
|---|---|---|
| Operating system | Windows 10/11, 64-bit | Windows 11, 64-bit |
| Python | 64-bit Python 3.10 or 3.11 | Python 3.11 |
| CPU | Modern 64-bit Intel/AMD processor; plan for at least 4 cores | 8 or more cores, especially for CPU processing |
| System RAM | Plan for 16 GB with Low VRAM Mode and short chunks | 32 GB or more for long recordings and larger UVR models |
| GPU | Optional: CPU mode is supported | Compatible NVIDIA GPU with 8 GB or more VRAM; larger models may need more |
| Free disk space before full setup | Estimate **25–30 GiB** for the full CUDA installation | **30–35 GiB** on an SSD provides more setup headroom; budget recording/output storage separately |
| Media tools | FFmpeg and ffprobe on PATH, or configured executable paths | Both from the same FFmpeg installation |
| Browser | A browser that can access the local Gradio interface | Current Edge, Chrome or Firefox |
| Internet | Required for installing packages and downloading missing models/configuration files | Not required for processing once the selected models and metadata are cached |

**Tested GPU:** NVIDIA GeForce RTX 3060 Ti with 8 GB VRAM. ClearVoice, Resemble Enhance and UVR inference were exercised on this GPU; ClearVoice and Resemble were also exercised in CPU mode. These short integration tests do not establish a minimum-memory requirement or guarantee that every UVR model will fit. Start with Low VRAM Mode enabled and reduce chunks to 5–15 seconds if needed. CPU processing speed depends strongly on the model and processor; real-time processing is not guaranteed.

The current setup installs PyTorch 2.5.1 with CUDA 12.1 when NVIDIA drivers are detected. The GPU architecture and driver must support that installed build; detection of an NVIDIA card alone is insufficient. A separate CUDA toolkit is not required for the prebuilt wheels. AMD/Intel GPUs are not configured for acceleration by this launcher; use CPU mode. See the [official PyTorch wheel combinations](https://pytorch.org/get-started/previous-versions/#v251) and [NVIDIA driver compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

### Installed disk usage

Measured in this development installation on **2026-09-23**, using the sum of file sizes (not filesystem “size on disk”). **1 GiB = 1,073,741,824 bytes**; decimal GB values are larger numerically.

| Folder | Measured size |
|---|---:|
| `.venv` — UI and orchestration | 0.55 GiB |
| `.venv_clearvoice` | 4.88 GiB |
| `.venv_resemble` | 4.88 GiB |
| `.venv_uvr` | 5.55 GiB |
| **All four environments** | **15.84 GiB (17.01 GB)** |
| `models/` — downloaded weights and local model caches | 1.35 GiB |
| `temp/` — current temporary files and previews | 0.46 GiB |
| `outputs/` — current generated results | 0.34 GiB |
| **Entire project folder, including the above** | **17.99 GiB (19.32 GB)** |

Totals use unrounded byte counts. This is a snapshot, not a fixed installation size: additional UVR models, private copies of UVR5 weights, recordings and generated results increase it. Pip's download cache and installation staging can also consume several gigabytes **outside this project folder**, so the measured total is not the full amount of disk space needed during setup.

The **25–30 GiB setup estimate** allows for the roughly 18 GiB installed workspace plus package downloads and temporary installation files. **30–35 GiB** provides a more comfortable margin. These are planning estimates, not measured peak-installation requirements or enforced limits; 40–60 GiB is not required just to install the application. A CPU-only or UI-only installation can be smaller, but those installation sizes have not been measured here.

Most of the installed size comes from separate PyTorch/CUDA dependencies in the three AI environments. Their isolation avoids conflicting package requirements; merging them to save space is not supported. Low VRAM Mode reduces GPU memory use, **not disk usage**.

### Space for processing recordings

- One hour of 48 kHz float WAV uses approximately **0.64 GiB mono** or **1.29 GiB stereo per intermediate file**. Several stage outputs coexist until the job finishes, even when saving intermediates is disabled.
- As a conservative working budget, allow an additional **10 GiB per hour of mono audio**, or **20 GiB per hour of stereo audio**, when using multiple AI stages. This is headroom, not a measured maximum.
- Add space for the input/upload cache and remuxed video separately; a large video can dominate the total. Gradio's preview/download cache can create additional media copies. Keeping intermediates or processing multiple jobs grows retained storage further.
- Clean Temporary Files removes application-owned temporary job folders when idle. It does not remove outputs, model caches or virtual environments.

### Source repository versus installed application

The **17.99 GiB folder is the installed workspace**, not the size of the source code to publish. `.gitignore` already excludes `.venv*`, downloaded model weights/caches, outputs, logs and temporary files. Distribute the source, launcher, requirements, configuration and documentation; each computer runs Setup to create its own environments. Avoid ZIPping the entire installed folder for a source release, and do not treat copied virtual environments as a portable Windows installer.

## Start here

1. Install **64-bit Python 3.10 or 3.11** and enable **Add Python to PATH**.
2. Install **FFmpeg with ffprobe**. On Windows: `winget install Gyan.FFmpeg`. Reopen the launcher after installing it. Alternatively, set both executable paths in `config.json`.
3. Double-click **VoiceRestore.bat**, then choose **1 — Setup / Repair Environment**. Leave the terminal open; PyTorch downloads are large. Setup prints installation output and saves `logs/setup.log`.
4. Choose **2 — Run VoiceRestore Studio**. Your browser opens on `http://127.0.0.1:7860`, or the next free port.
5. On first use, expand **Model Status & Maintenance** and select **Download / Verify Models**. ClearVoice and Resemble weights are cached after their first successful download. For later use no network access is required.
6. Drop a recording, choose **Auditorium Speech**, optionally toggle restoration stages, and press **PROCESS RECORDING**.

Results appear in `outputs/<timestamp>_<name>_<job-id>/`. Download the final WAV and FLAC (audio input), or WAV and remuxed video (video input), plus the JSON/text processing reports and job log. **Open Output Folder** opens that job's directory. Close the server terminal with Ctrl+C when done; the launcher never kills unrelated Python processes.

**All processing is performed locally on this computer.** No accounts, analytics, cloud inference or outgoing media uploads. Only setup and explicit model downloads require external services.

## Pipeline and listening

`FFmpeg → optional UVR vocal isolation → ClearVoice → optional UVR de-reverb → optional Resemble → DSP → loudness → export`

- ClearVoice and Resemble default on. UVR stages default off. Each stage is independent.
- Natural Speech applies gentler EQ/restoration; Auditorium Speech is the default; Podcast Clear and Strong Cleanup provide alternatives. All controls remain editable. Reports contain the actual settings, not just a preset label.
- More processing is not always better. Strong restoration can change the voice or consonants. Listen to the original and intermediate players before increasing strength.
- Resemble's **Lambda** is its internal denoiser-conditioning mix, not a generic wet/dry enhancement slider. Denoise-before-enhancement is separate and defaults off to avoid redundant denoising after ClearVoice.
- Auto channel handling normally mixes to mono, but samples the start, middle and end to avoid cancellation from opposing-phase channels or a nearly silent channel. This is a conservative signal heuristic, not a speech-quality classifier. Manual left/right selection is available.
- Keep Stereo runs speech models per channel. The stereo image can change; mono is usually better for auditorium speech.
- Original and intermediate previews are bounded to the first 60 seconds unless **Save intermediate files** is on. The final master is available in full. Waveforms display the first 60 seconds and cannot block an otherwise successful export.

## UVR de-reverb and vocal models

UVR uses **python-audio-separator**, not the desktop UVR application. Place compatible, trusted `.pth`, `.onnx` or `.ckpt` weights and their configuration files in `models/uvr`, or paste another local folder under Advanced Settings. Click **Scan Model Directory**. Filename heuristics distinguish echo/reverb candidates from other models; the engine validates actual support when loading.

**Already have UVR5?** Paste its top-level `models` directory. The scanner searches `MDX_Net_Models`, `VR_Models` and other subfolders recursively. Select a vocal model (for example `MDX_Net_Models/Kim_Vocal_2.onnx`), leave **Vocal stem name** as `Vocals`, and enable **UVR Vocal Isolation**. Studio's own de-reverb models still appear with an `@studio/` prefix. Denoiser-only weights and raw Demucs checkpoints are excluded from the vocal list.

Existing UVR5 weights and matching `model_data`/MDXC YAML files are copied once into VoiceRestore's private cache, without modifying the UVR5 installation or downloading those weights again. Companion catalogs/configuration files can be fetched with **Download / Verify Models → uvr** if missing. Restart VoiceRestore after updating application code to get the recursive scanner.

Examples include MelBand Roformer de-reverb/de-echo variants and UVR-DeEcho-DeReverb. The application does not depend on an exact filename. Renaming a model can break upstream catalog lookup; preserve its original name. Arbitrary custom architectures are not automatically supported.

Select the **exact dry stem name** from your model card (such as `No Reverb`, `No Echo`, or `Vocals`). The app deliberately fails if the requested stem is missing or ambiguous. Use **Download / Verify Models → uvr** once to cache companion catalog/configuration files. Local weights are required; the application does not pick or purchase a model for you.

Aggressiveness is a dry/wet blend with the selected de-reverbed stem: Conservative 55%, Balanced 75%, Strong 100%. Vocal isolation is a separate pre-enhancement stage and is intended only for strong music or competing non-speech audio.

## Devices, long recordings and cancellation

- The worker validates CUDA by allocating and running a tensor operation. CPU is used if CUDA is unavailable or unsupported; explicit CPU selection is respected.
- **Low VRAM Mode** (default on) closes each isolated worker after its stage, releasing all its GPU allocations. With it off, workers and models are reused across jobs. **Unload Models** releases them when idle.
- AI processing uses bounded chunks (30 seconds, 1-second overlap by default), linear crossfades, and disk-backed WAV files. It never sends a whole multi-hour recording to GPU memory. Reduce chunks to 5–15 seconds for memory-heavy models.
- Only one processing job/download runs at once. STOP signals cancellation, terminates the active FFmpeg/model worker, discards partial exports and preserves a cancellation report. It does not terminate other applications.
- Temporary job folders use UUIDs. Completed jobs discard full intermediates unless requested; short preview files remain until **Clean Temporary Files** is used. Gradio maintains a separate preview cache. Outputs and sources are never removed by this button.
- Allow sufficient disk space: the measured full installation is about 18 GiB before substantial recording storage. See **System requirements** above for setup headroom and per-hour processing budgets.

## Mastering and export

Internal audio is float WAV, 48 kHz. Final WAV is 24-bit PCM. Mastering provides a two-pole 80 Hz high-pass, gentle 280/650 Hz cuts and 3/5 kHz boosts, 2.5:1 compression around −18 dB (10 ms attack / 120 ms release), and a −1 dBTP limiter target with 4× oversampling. Two-pass FFmpeg `loudnorm` targets −16 LUFS. Actual final integrated loudness, loudness range and true peak are measured and reported; targets are never presented as measurements. Silent audio remains silent.

Video packets are copied, with processed audio encoded as 256 kbps AAC. MP4 is preferred; if the input video codec cannot be muxed into MP4, MKV is used without re-encoding video. The selected video stream and first audio stream are retained; alternate audio tracks/subtitles are not included. Audio/video start-time offsets are preserved. Original duration and processing duration are recorded.

Reports include every requested setting, enabled/disabled/completed/failed stages, model package versions, device, model weight SHA256 fingerprints, output files and final measurements. Failed stages are never silently bypassed. Full tracebacks are kept in logs, with actionable messages in the interface.

## Environment design

| Environment | Purpose | Main versions |
|---|---|---|
| `.venv` | UI, FFmpeg orchestration, streaming DSP | Gradio 5.49.1, NumPy 1.26.4 |
| `.venv_clearvoice` | Speech enhancement worker | ClearVoice 0.1.2, Torch 2.5.1 |
| `.venv_resemble` | Restoration worker | Resemble 0.0.1 + documented inference adaptation, Torch 2.5.1 |
| `.venv_uvr` | UVR-compatible inference worker | audio-separator 0.30.2, Torch 2.5.1 |

Setup chooses official PyTorch CUDA 12.1 wheels when NVIDIA drivers are detected, otherwise CPU wheels. A system CUDA toolkit is not required. Older drivers/unsupported GPU architectures may require updating the driver or installing CPU wheels: `python scripts/setup_env.py --cpu`. You can repair one engine with `--engine clearvoice`, `--engine resemble` or `--engine uvr`; use `--core-only` for the UI and DSP.

The released Resemble package pins old Gradio/Torch and imports DeepSpeed training code from inference. `scripts/build_resemble.py` builds a **SHA256-verified upstream wheel** into an explicitly versioned local inference-only wheel (`0.0.1+voicerestore.1`). It replaces training-module imports with direct model imports and removes training-only visualization; model networks, solvers, audio inference and upstream license notices remain intact. It does not stub DeepSpeed or simulate inference. Training is unsupported in that environment. The patch fails if the expected upstream inference import has changed.

Every environment runs `pip check` and an import health check. Exact resolved installations are saved to `logs/venv*-resolved.txt`. The application uses persistent subprocess RPC so conflicting model dependencies cannot alter Gradio's environment.

The UVR adapter also corrects the pinned upstream version's WAV export path and float-stereo handling. It writes the genuine separated samples directly with SoundFile and resamples the model's native 44.1 kHz output back to the pipeline's 48 kHz. Resemble's Linux path tags are read with a restricted, portable YAML loader on Windows.

## Configuration and troubleshooting

`config.json` contains defaults for FFmpeg/ffprobe paths, output/model directories, device, port and preferences. UI preferences are saved atomically in the ignored `config.local.json` override. Paths can be absolute or relative to the project; no user-specific paths are hardcoded.

- **Missing model**: use Download / Verify Models. A cache status means that the model was loaded successfully previously; weights are checked again at load.
- **Out of memory**: turn on Low VRAM Mode, reduce chunk size, disable the failing stage, or choose CPU.
- **UVR unsupported/missing stem**: check the original filename, architecture support, matching YAML and exact dry stem label.
- **No sound in media**: check that the input actually contains an audio stream; silent inputs remain silent.
- **Setup failure**: read `logs/setup.log`; rerun Setup / Repair after addressing the first pip/import error. Ensure adequate free space and a 64-bit Python 3.10/3.11 install.
- **Port in use**: the app searches the next 100 localhost ports and prints the chosen URL.

Model weights retain their upstream licenses and usage terms. Use trusted weights: legacy PyTorch checkpoints may execute Python objects when loaded. Perceptual restoration cannot recover the exact original speech or guarantee identity/transcript fidelity.

## Development and verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts app.py tests
.\.venv\Scripts\python.exe app.py --no-browser
# In another terminal, exercise actual Gradio uploads/callbacks/downloads:
.\.venv\Scripts\python.exe scripts/verify_ui.py
# Opt-in real model download + synthetic speech GPU inference:
.\.venv\Scripts\python.exe scripts/verify_models.py --download
```

Tests cover sample-exact identity chunking, stereo phase handling, real audio/video exports, video-bitstream preservation, measured loudness, silence, source preservation, prompt subprocess cancellation, and failed-stage reporting. Model tests use real upstream inference and are intentionally separate from the small automated suite.

Installation/API references checked during implementation:

- [ClearerVoice Studio installation and public API](https://github.com/modelscope/ClearerVoice-Studio/blob/main/clearvoice/README.md)
- [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) and [released package](https://pypi.org/project/resemble-enhance/0.0.1/)
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks)
