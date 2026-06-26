param (
    [string]$InnoSetupCompiler = "",
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"

function Resolve-InnoSetupCompiler {
    <#
    .SYNOPSIS
    Resolve the Inno Setup command line compiler path.
    #>
    param (
        [string]$CompilerPath
    )

    if ($CompilerPath) {
        $resolvedCompilerPath = Resolve-Path -Path $CompilerPath -ErrorAction Stop
        return $resolvedCompilerPath.Path
    }

    $command = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    [string[]]$candidateRoots = @(
        ${env:ProgramFiles(x86)},
        $env:ProgramFiles
    )
    foreach ($candidateRoot in $candidateRoots) {
        if (-not $candidateRoot) {
            continue
        }
        $candidatePath = Join-Path -Path $candidateRoot -ChildPath "Inno Setup 6\ISCC.exe"
        if (Test-Path -Path $candidatePath -PathType Leaf) {
            return $candidatePath
        }
    }

    throw "Could not find ISCC.exe. Install Inno Setup 6 or pass -InnoSetupCompiler."
}

function Get-SoftwareVersion {
    <#
    .SYNOPSIS
    Return the image2image version from the active Python environment.
    #>
    param (
        [string]$Executable
    )

    [string[]]$versionOutput = & $Executable -c "import image2image; print(image2image.__version__)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        [string]$errorText = $versionOutput -join [Environment]::NewLine
        throw "Could not read image2image.__version__ using '$Executable': $errorText"
    }
    if ($versionOutput.Count -eq 0) {
        throw "Could not read image2image.__version__ using '$Executable': command produced no output."
    }
    return $versionOutput[-1].Trim()
}

function Invoke-InnoSetupCompiler {
    <#
    .SYNOPSIS
    Compile the image2image Inno Setup installer.
    #>
    param (
        [string]$CompilerPath,
        [string]$ScriptPath,
        [string]$Version,
        [string]$SourceDir,
        [string]$OutputDir,
        [string]$IconFile
    )

    [string]$outputBaseFilename = "image2image-$Version-win_amd64-setup"
    [string[]]$arguments = @(
        "/DAppName=image2image",
        "/DAppExeName=image2image.exe",
        "/DAppVersion=$Version",
        "/DAppPublisher=vandeplaslab",
        "/DSourceDir=$SourceDir",
        "/DOutputDir=$OutputDir",
        "/DOutputBaseFilename=$outputBaseFilename",
        "/DIconFile=$IconFile",
        $ScriptPath
    )

    & $CompilerPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup compiler failed with exit code $LASTEXITCODE."
    }

    $installerPath = Join-Path -Path $OutputDir -ChildPath "$outputBaseFilename.exe"
    if (-not (Test-Path -Path $installerPath -PathType Leaf)) {
        throw "Expected installer was not created: $installerPath"
    }

    # Code signing hook: sign $installerPath here when a signing certificate is configured.
    return $installerPath
}

[string]$scriptRoot = $PSScriptRoot
[string]$repoRoot = Split-Path -Path (Split-Path -Path $scriptRoot -Parent) -Parent
[string]$distDir = Join-Path -Path $scriptRoot -ChildPath "dist"
[string]$sourceDir = Join-Path -Path $distDir -ChildPath "image2image"
[string]$appExe = Join-Path -Path $sourceDir -ChildPath "image2image.exe"
[string]$issFile = Join-Path -Path $scriptRoot -ChildPath "installer.iss"
[string]$iconFile = Join-Path -Path $repoRoot -ChildPath "src\image2image\assets\icon.ico"

if (-not (Test-Path -Path $appExe -PathType Leaf)) {
    throw "Could not find PyInstaller output executable: $appExe"
}
if (-not (Test-Path -Path $issFile -PathType Leaf)) {
    throw "Could not find Inno Setup script: $issFile"
}
if (-not (Test-Path -Path $iconFile -PathType Leaf)) {
    throw "Could not find installer icon: $iconFile"
}

[string]$compilerPath = Resolve-InnoSetupCompiler -CompilerPath $InnoSetupCompiler
[string]$version = Get-SoftwareVersion -Executable $PythonExecutable
[string]$installerPath = Invoke-InnoSetupCompiler `
    -CompilerPath $compilerPath `
    -ScriptPath $issFile `
    -Version $version `
    -SourceDir $sourceDir `
    -OutputDir $distDir `
    -IconFile $iconFile

Write-Output "Created installer: $installerPath"
