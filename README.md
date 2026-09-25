# AutoClip

**Open-source, local-first AI video clipper.** Long video in → ranked, caption-burned, speaker-tracked 9:16 clips out. A fork of [artbyjazi/autoclip](https://github.com/artbyjazi/autoclip) with extra features.

[![CI](https://github.com/sandeepannandi/AutoClip/actions/workflows/ci.yml/badge.svg)](https://github.com/sandeepannandi/AutoClip/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)

Paste a YouTube link or drop a file. AutoClip transcribes it, an LLM finds the moments worth clipping, it reframes them to vertical while tracking the speaker, burns in animated captions, and exports platform-ready MP4s. No accounts, no uploads to anyone's servers, no watermarks, no subscription.

---

## This fork

Everything the original [autoclip](https://github.com/artbyjazi/autoclip) ships, plus:

- **Color grading you can actually see** — four presets (`warm`, `punchy`, `cool`, `film`) with a live CSS preview and a _Compare with original footage_ toggle.
- **Per-clip caption colour** — override the caption text colour per clip; the preview and the burnt-in export match.
- **Captions never overlap** — wrapped lines are clamped so long runs of words can't collide.
- **Calmer speaker tracking that stays centred** — a lazy-follow (hysteresis dead-band) controller parks the camera until a genuinely big move, then keeps settling toward the speaker at an invisible ~1 px/s, so the resting frame is centred instead of frozen wherever the last pan stopped. Close-ups are tracked rather than mean-locked, and locks sit on the median position.
- **Performance feedback loop** — log where you posted a clip (`autoclip track post`), log its stats over time (`autoclip track stats`), and the ranker learns from your account's own results: per-platform baselines, outperformance-based re-ranking, and few-shot prompt examples from your best and worst performers.
- **Hardened highlights & ingestion** — real per-window errors, robust model-output coercion, and Windows `update-ytdlp`/cookie-database fixes.

## Quickstart

```bash
git clone https://github.com/sandeepannandi/AutoClip.git
cd autoclip && uv venv --python 3.11 && uv pip install -e ".[dev]"
cd frontend && npm install && npm run build && cd ..
autoclip doctor
autoclip serve    # opens http://localhost:8000
```

No [uv](https://docs.astral.sh/uv/)? `python -m venv .venv && pip install -e ".[dev]"` works the same.

## Requirements

- **Python 3.11 or 3.12** — not 3.13 (MediaPipe ships no 3.13 wheels).
- **ffmpeg** — a full build with `libass` and `libx264`, both on PATH.
- **Node 20+** to build the UI (not needed at runtime).
- **GPU** — optional; NVIDIA or Apple Silicon speed up transcription.

`autoclip doctor` checks all of this and prints the fix for anything missing.

### Installing ffmpeg

Captions are burned in with **libass**, which the obvious package omits on two platforms:

- **macOS** — `brew install ffmpeg-full`. Already on the plain formula? `brew unlink ffmpeg && brew link --force --overwrite ffmpeg-full`.
- **Windows** — `winget install Gyan.FFmpeg` (the full build, not `Gyan.FFmpeg.Essentials`).
- **Linux** — the distro package is fine: `sudo apt install ffmpeg`.

## Add a provider

Clip selection needs a language model. Paste a key in **Settings → Keys**, or run `autoclip config set-secret anthropic`. Keys live in your OS keyring, never a config file. Fully local instead? Install [Ollama](https://ollama.com) and `ollama pull llama3.1:8b`.

Privacy: only transcript _text_ is ever sent to a provider — never video or audio. With Ollama, nothing leaves the machine.

### Optional extras

```bash
uv pip install -e ".[gpu]"          # CUDA libs for NVIDIA GPUs — if doctor says CTranslate2 can't see yours
uv pip install -e ".[diarization]"  # WhisperX speaker diarization (pulls PyTorch; needs a HuggingFace token)
```

### Docker

```bash
docker compose -f docker/compose.yaml up --build
docker compose -f docker/compose.yaml --profile gpu up --build
```

## Using it

Everything in the UI is also on the CLI: `doctor` (check), `serve` (web app), `clip <url|file>` (full pipeline), `jobs`, `providers`, `styles`, `track` (posting performance), `config show`, `update-ytdlp`.

Caption styles: `bold_pop` (chunky, word lights up), `karaoke_fill` (words fill as spoken), `clean_lower` (minimal), `boxed` (high contrast).

## Troubleshooting

- **Says Python is wrong / you're on 3.13** — `uv venv --python 3.11`.
- **"cublas64_12.dll is not found"** — `uv pip install -e ".[gpu]"`.
- **"No such filter: ass" or captions missing** — your ffmpeg lacks libass; reinstall per [Installing ffmpeg](#installing-ffmpeg).
- **YouTube downloads hit a bot check** — set **Settings → Ingest → cookies from browser** and close that browser first (it locks its cookie DB while running).
- **Exports slower than expected** — check `doctor` for GPU encoding; AutoClip probes NVENC and falls back to CPU when the driver can't use it.

## How it works

`ingest → prepare → transcribe → highlights → reframe → captions → export`

Each stage writes to `~/.autoclip/work/{job_id}/`, so retries resume where they failed. Highlights return word indices, not timestamps (models are bad at arithmetic but good at copying), and crop paths never pan across a cut. Details in [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Testing

```bash
uv run pytest -m "not slow and not golden and not e2e" -q   # unit tests
uv run pytest -m "slow and not golden and not e2e" -q       # real ffmpeg render tests
AUTOCLIP_E2E_MEDIA=/path/to/clip.mp4 uv run pytest -m e2e   # full pipeline on real footage
```

## Contributing

Prompts and caption styles are the highest-leverage places to start and need no deep knowledge of the codebase. See [CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Legal & License

Only download content you own or have the rights to process; no DRM or paywall workarounds, ever. MIT — see [LICENSE](LICENSE). Bundled fonts (Anton, Inter) are under the SIL Open Font License.
