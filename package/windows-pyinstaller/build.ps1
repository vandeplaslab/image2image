param (
    [switch]$activate,
    [switch]$update,
    [switch]$update_app,
    [switch]$update_deps,
    [switch]$update_just_app,
    [switch]$update_just_reader,
    [switch]$update_just_register,
    [switch]$update_pip,
    [switch]$update_qt,
    [switch]$no_build,
    [switch]$debug,
    [switch]$zip,
    [switch]$installer,
    [string]$InnoSetupCompiler = "",
    [switch]$run,
    [switch]$help
)

$ErrorActionPreference = "Stop"

if ($help) {
    Write-Output @"
Usage: build.ps1 [-update] [-update_app] [-update_deps] [-update_pip] [-update_qt]
  -update: Refresh all local and registry dependencies.
  -update_app: Refresh image2image, image2image-io, and image2image-reg.
  -update_deps: Refresh local ecosystem dependencies and native backends.
  -update_pip: Upgrade registry dependencies within project requirements.
  -update_qt: Refresh qtextra and qtextraplot.
  -update_just_app / -update_just_reader / -update_just_register: Refresh one project.
  -no_build: Set up/update the environment without building.
  -activate: Create and activate the environment, then return (dot-source to retain activation).
  -debug: Enable PyInstaller debug output.
  -zip: Create a versioned ZIP.
  -installer: Compile an Inno Setup installer.
  -InnoSetupCompiler: Optional path to ISCC.exe.
  -run: Launch the built application.
  -help: Show help.

Creates venv_package with uv and Python 3.12 when missing.
Set PACKAGING_PYTHON to override the Python version for a new environment.
"@
    return
}

[string]$repo_dir = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
[string]$venv_dir = Join-Path $PSScriptRoot "venv_package"
[string]$python = Join-Path $venv_dir "Scripts\python.exe"
[string[]]$install_args = @()
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it before running this script."
}
if (-not (Test-Path $venv_dir)) {
    $python_version = if ($env:PACKAGING_PYTHON) { $env:PACKAGING_PYTHON } else { "3.12" }
    & uv venv --python $python_version $venv_dir
    if ($LASTEXITCODE -ne 0) { throw "uv could not create the packaging environment." }
    $install_args += "--bootstrap"
}
if (-not (Test-Path $python -PathType Leaf)) {
    throw "Invalid environment at $venv_dir. Rename it and rerun this script."
}
if ($activate) {
    . (Join-Path $venv_dir "Scripts\Activate.ps1")
    return
}

foreach ($flag in @("update", "update_app", "update_deps", "update_pip", "update_qt", `
                    "update_just_app", "update_just_reader", "update_just_register")) {
    if ((Get-Variable -Name $flag -ValueOnly)) {
        $install_args += "--$($flag.Replace('_', '-'))"
    }
}

$old_qt_api = $env:QT_API
$old_debug = $env:PYINSTALLER_DEBUG
Push-Location $PSScriptRoot
try {
    $env:QT_API = "pyqt6"
    $env:PYINSTALLER_DEBUG = if ($debug) { "all" } else { "0" }
    & $python (Join-Path $repo_dir "package\install_dependencies.py") @install_args
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation or validation failed with exit code $LASTEXITCODE." }
    if ($no_build) { return }

    & $python (Join-Path $repo_dir "scripts\print_versions.py")
    if ($LASTEXITCODE -ne 0) { throw "Printing package versions failed with exit code $LASTEXITCODE." }
    & $python -m PyInstaller --noconfirm --clean (Join-Path $PSScriptRoot "image2image.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE." }
    if ($zip) {
        & $python (Join-Path $PSScriptRoot "zip.py")
        if ($LASTEXITCODE -ne 0) { throw "ZIP creation failed with exit code $LASTEXITCODE." }
    }
    if ($installer) {
        & (Join-Path $PSScriptRoot "installer.ps1") -PythonExecutable $python -InnoSetupCompiler $InnoSetupCompiler
    }
    if ($run) {
        Start-Process (Join-Path $PSScriptRoot "dist\image2image\image2image.exe") `
            -WorkingDirectory (Join-Path $PSScriptRoot "dist\image2image")
    }
} finally {
    Pop-Location
    $env:QT_API = $old_qt_api
    $env:PYINSTALLER_DEBUG = $old_debug
}
