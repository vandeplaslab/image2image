# Archive an existing build using its packaging environment.
$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "venv_package\Scripts\python.exe"
if (-not (Test-Path $python -PathType Leaf)) {
    throw "Create the packaging environment with build.ps1 first."
}
& $python (Join-Path $PSScriptRoot "zip.py")
if ($LASTEXITCODE -ne 0) { throw "ZIP creation failed with exit code $LASTEXITCODE." }
