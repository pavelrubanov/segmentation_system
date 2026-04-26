# Build Windows release via PyInstaller (--onedir).
#
# Usage: .\packaging\build_windows.ps1
#
# Requires: Python 3.10 available via `py -3.10`.
# Output:   dist/sphagnum-cpu-win64/

$ErrorActionPreference = 'Stop'

$BuildDir    = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $BuildDir
$SrcDir      = Join-Path $ProjectRoot 'src'
$DistRoot    = Join-Path $ProjectRoot 'dist'
$SpecFile    = Join-Path $BuildDir 'sphagnum.spec'
$ReqFile     = Join-Path $SrcDir 'requirements.txt'
$Venv        = Join-Path $BuildDir '.venv-cpu'
$WorkDir     = Join-Path $BuildDir 'work-cpu'
$VenvPython  = Join-Path $Venv 'Scripts\python.exe'

if (-not (Test-Path $Venv)) {
    Write-Host "Creating venv: $Venv"
    & py '-3.10' -m venv $Venv
}

# python -m pip/PyInstaller (а не Scripts\*.exe-обёртки): обёртки на Windows
# содержат хардкод-путь к python.exe и ломаются, если venv переезжает.
& $VenvPython -m pip install --upgrade pip wheel setuptools
& $VenvPython -m pip install -r $ReqFile
& $VenvPython -m pip install 'pyinstaller==6.11.1'

$DefaultDist = Join-Path $DistRoot 'sphagnum'
$FinalDist   = Join-Path $DistRoot 'sphagnum-cpu-win64'
if (Test-Path $WorkDir)     { Remove-Item -Recurse -Force $WorkDir }
if (Test-Path $DefaultDist) { Remove-Item -Recurse -Force $DefaultDist }
if (Test-Path $FinalDist)   { Remove-Item -Recurse -Force $FinalDist }

& $VenvPython -m PyInstaller --noconfirm --clean `
    --workpath $WorkDir --distpath $DistRoot $SpecFile

Rename-Item -Path $DefaultDist -NewName 'sphagnum-cpu-win64'
Write-Host "Done: $FinalDist" -ForegroundColor Green
