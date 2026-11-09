# Compatibility entry point for the unified packaging environment setup.
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "..\package\windows-pyinstaller\build.ps1") -no_build @args
