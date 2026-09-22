# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the AutoClip Windows executable (onedir, CPU-only).

Build from the repository root:
    pyinstaller packaging/autoclip.spec --noconfirm

Layout produced under dist/AutoClip/:
    AutoClip.exe
    _internal/            Python bundle, collected packages, autoclip data
        ffmpeg/           bundled ffmpeg.exe + ffprobe.exe (prepended to PATH
                          by packaging/launcher.py when frozen)
        autoclip/static/  compiled React SPA
        autoclip/assets/  caption fonts
        autoclip/prompts/ versioned prompt files
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is injected
BACKEND = ROOT / "backend"

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

datas: list[tuple[str, str]] = []

# Compiled frontend (index.html + assets/) — must land at
# autoclip/static so app.py's Path(__file__).parent / "static" resolves.
static_index = BACKEND / "autoclip" / "static" / "index.html"
if not static_index.exists():
    raise SystemExit(
        "backend/autoclip/static/index.html is missing. Build the frontend "
        "first: cd frontend && npm install && npm run build"
    )
datas.append((str(BACKEND / "autoclip" / "static"), "autoclip/static"))

# Caption fonts (pipeline/captions.py: Path(__file__).parent.parent / assets/fonts)
datas.append((str(BACKEND / "autoclip" / "assets"), "autoclip/assets"))

# Versioned prompt files (providers/base.py: ... / prompts)
datas.append((str(BACKEND / "autoclip" / "prompts"), "autoclip/prompts"))

# Bundled ffmpeg/ffprobe (launcher prepends this dir to PATH when frozen).
ffmpeg_dir = ROOT / "packaging" / "ffmpeg"
for exe in ("ffmpeg.exe", "ffprobe.exe"):
    if not (ffmpeg_dir / exe).exists():
        raise SystemExit(
            f"packaging/ffmpeg/{exe} is missing. Copy a full ffmpeg build "
            "(with libass + libx264) there before building."
        )
datas.append((str(ffmpeg_dir), "ffmpeg"))

# Application icon — the favicon from frontend/index.html, rendered to a
# multi-size .ico by packaging/make_icon.py.
icon_path = ROOT / "packaging" / "icon.ico"
if not icon_path.exists():
    raise SystemExit(
        "packaging/icon.ico is missing. Generate it with: "
        "python packaging/make_icon.py packaging/icon.ico"
    )

# --- Package data that import-time code resolves via __file__ -------------
# mediapipe: libmediapipe.dll + label maps / metadata schemas
datas += collect_data_files("mediapipe")
# faster-whisper: silero VAD onnx asset
datas += collect_data_files("faster_whisper")
# onnxruntime: DLLs are collected by the contrib hook; nothing data-side here.

# --- Distribution metadata (entry-point / version lookups at runtime) ------
# keyring discovers backends via entry points (official hook also does this;
# copying twice is harmless and keeps us safe if the hook is skipped).
datas += copy_metadata("keyring")
# cli.py's update-ytdlp reads importlib.metadata.version("yt-dlp").
datas += copy_metadata("yt-dlp")
# faster-whisper / huggingface internals may probe installed distributions.
datas += copy_metadata("faster_whisper")

# ---------------------------------------------------------------------------
# Hidden imports (imported by string, or only reached behind a frozen guard)
# ---------------------------------------------------------------------------

hiddenimports: list[str] = [
    # uvicorn resolves loops/protocols/lifespan by dotted string.
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.websockets.websockets_sansio_impl",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    # MediaPipe FaceLandmarker (imported inside a function in reframe/faces.py)
    "mediapipe.tasks.python.core.base_options",
    "mediapipe.tasks.python.vision.face_landmarker",
    "mediapipe.tasks.python.vision.core.vision_task_running_mode",
    # faster-whisper is imported lazily in pipeline/transcribe.py
    "faster_whisper",
    "faster_whisper.assets",
    # yt-dlp is a large flat package; collect everything to be safe.
    *collect_submodules("yt_dlp"),
    # keyring backends (Windows Credential Locker etc.)
    *collect_submodules("keyring.backends"),
    # scenedetect backend detection
    "scenedetect.backends.opencv",
    # Google GenAI / provider SDKs import submodules lazily.
    "google.genai",
    "google.genai.types",
    "anthropic",
    "openai",
    # pysubs2 uses pkg resources for style parsing in some paths
    "pysubs2",
    # sse-starlette eventsource responses
    "sse_starlette.sse",
]

# ---------------------------------------------------------------------------
# Binary libraries (DLLs living beside pure-Python packages)
# ---------------------------------------------------------------------------

# ctranslate2: ctranslate2.dll, libiomp5md.dll (+ optional cudnn stub)
binaries = collect_dynamic_libs("ctranslate2")
# mediapipe: tasks/c/libmediapipe.dll
binaries += collect_dynamic_libs("mediapipe")
# onnxruntime DLLs (contrib hook also covers this; belt and braces)
binaries += collect_dynamic_libs("onnxruntime")

# ---------------------------------------------------------------------------
# Excludes — keep the bundle CPU-only and lean
# ---------------------------------------------------------------------------

excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "whisperx",
    "pyannote",
    "nvidia",  # CUDA wheels must never leak into the CPU build
    "tkinter",
    "matplotlib",
    "IPython",
    "pytest",
]

# ---------------------------------------------------------------------------
# Analysis / EXE / COLLECT
# ---------------------------------------------------------------------------

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(BACKEND), str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoClip",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    icon=str(icon_path),
    console=True,  # keep a terminal for CLI output and server logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AutoClip",
)
