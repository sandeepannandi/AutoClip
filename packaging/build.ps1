$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$checks = @(
    @{ Path = "backend\autoclip\static\index.html"; Message = "Frontend not built. Run `cd frontend && npm install && npm run build` first." },
    @{ Path = "packaging\ffmpeg\ffmpeg.exe";  Message = "ffmpeg.exe missing. Copy a full ffmpeg build into packaging\ffmpeg\." },
    @{ Path = "packaging\ffmpeg\ffprobe.exe"; Message = "ffprobe.exe missing. Copy a full ffmpeg build into packaging\ffmpeg\." },
    @{ Path = "packaging\icon.ico";               Message = "icon.ico missing. Generate it with `python packaging\make_icon.py packaging\icon.ico`." }
)

foreach ($c in $checks) {
    if (-not (Test-Path -LiteralPath $c.Path)) {
        throw $c.Message
    }
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment not found at .venv. Create it and install dependencies first."
}

Write-Host "Building AutoClip with PyInstaller..."
& $VenvPython -m PyInstaller packaging\autoclip.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed (exit code $LASTEXITCODE)."
}

Write-Host "`nBuild complete: dist\AutoClip\AutoClip.exe"