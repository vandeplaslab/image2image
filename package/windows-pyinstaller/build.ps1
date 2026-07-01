param (
    [switch]$activate = $false,
    [switch]$update = $false,
    [switch]$update_app = $false,
    [switch]$update_deps = $false,
    [switch]$update_just_app = $false,
    [switch]$update_just_reader = $false,
    [switch]$update_just_register = $false,
    [switch]$update_pip = $false,
    [switch]$no_build = $false,
    [switch]$debug = $false,
    [switch]$zip = $false,
    [switch]$run = $false,
    [switch]$help = $false
)

if ($help) {
    Write-Output "Package app for Windows

    Output parameters:
    -activate: Activate environment. Default=False
    -update: Update all modules. Default=False
    -update_app: Update all modules. Default=False
    -update_deps: Update all dependencies. Default=False
    -update_just_app: Update just the app. Default=False
    -update_just_reader: Update just the reader. Default=False
    -update_just_register: Update just the register. Default=False
    -update_pip: Update pip packages. Default=False
    -debug: Debug mode. Default=False
    -zip: Compress distribution. Default=False
    -run: Run application after it has been built.
    -help: Print this message.
    "
    Exit
}
# print paths
$start_dir = $pwd
echo "Current directory: " $start_dir.Path
$github_dir = $start_dir | Split-Path | Split-Path | Split-Path
echo "Github directory: " $github_dir
$i2i_dir = Join-Path -Path $github_dir -ChildPath "image2image" -Resolve
# activate venv environment
$venv_activate = Join-Path -Path $i2i_dir -ChildPath "venv_package" | Join-Path -ChildPath "Scripts" | Join-Path -ChildPath "activate.ps1"
echo "Venv script location: " $venv_activate
& $venv_activate

# activate conda environment
if ($activate) {
    Exit
}

if ($debug) {
    echo "Debug mode enabled"
    $env:PYINSTALLER_DEBUG="all"
} else {
    $env:PYINSTALLER_DEBUG="0"
}

$python_ver = &{python -V} 2>&1
echo "Python version "$python_ver


# update all dependencies and app
if ($update) {
    $update_deps = $true
    $update_app = $true
    $update_pip = $true
}

# only update app
if ($update_app) {
    $update_just_app = $true
    $update_just_reader = $true
    $update_just_register = $true
}

# only update dependencies
$qtextra = $false
if ($update_deps) {
    $qtextra = $true
}

# update all dependencies and app
[System.Collections.ArrayList]$before_install = @()
[System.Collections.ArrayList]$pip_install = @()
[System.Collections.ArrayList]$after_install = @()
if ($update_pip) {
    $pip_install.Add("napari==0.6.6")
    $pip_install.Add("pydantic>=2")
    $pip_install.Add("pyqt6>=6.9.1")

    $after_install.Add("zarr>1,<3")
    $after_install.Add("tifffile<2025.5.10")
    $after_install.Add("pandas>2,<3")
    $after_install.Add("numpy>2")
    $after_install.Add("numba>0.60")
}

# always install latest version of pyinstaller
$pip_install.Add("pyinstaller")

# install before packages
foreach ($package in $before_install) {
    echo "Re-installing $package..."
    uv pip install -U $package
    echo "Reinstalled $package"
}

# install pip packages
foreach ($package in $pip_install) {
    echo "Re-installing $package..."
    uv pip install -U $package
    echo "Reinstalled $package"
}

[System.Collections.ArrayList]$local_install = @()
if ($update_just_app) {
    $local_install.Add("image2image")
}

if ($update_just_register) {
    $local_install.Add("image2image-reg")
}

if ($update_just_reader) {
    $local_install.Add("image2image-io")
}
$local_install.Add("koyo")

# install local packages
foreach ($package in $local_install) {
    echo "Re-installing $package..."
    $new_dir = Join-Path -Path $github_dir -ChildPath $package -Resolve
    cd $new_dir
    uv pip uninstall .
    uv pip install -U .
    cd $start_dir
    echo "Reinstalled $package"
}

# install qtextra
if ($qtextra) {
    echo "Re-installing qtextra..."
    $new_dir = Join-Path -Path $github_dir -ChildPath "qtextra" -Resolve
    cd $new_dir
    uv pip uninstall .
    uv pip install -U ".[sentry,console]"
    cd $start_dir
    echo "Reinstalled qtextra"

    echo "Re-installing qtextraplot..."
    $new_dir = Join-Path -Path $github_dir -ChildPath "qtextraplot" -Resolve
    cd $new_dir
    uv pip uninstall .
    uv pip install -U ".[2d]"
    cd $start_dir
    echo "Reinstalled qtextraplot"
}

# install after packages
foreach ($package in $after_install) {
    echo "Re-installing $package..."
    uv pip install -U $package
    echo "Reinstalled $package"
}

# uninstall pdbpp
uv pip uninstall pdbpp

if ($no_build) {
    Exit
}

# Get path
$filename = "image2image.spec"

# Build bundle
Write-Output "Filename: $filename"
pyinstaller.exe --noconfirm --clean $filename

if ($sign) {
    echo "Signing application executable..."
    [string[]]$sign_args = @()
    if ($sign_tool) {
        $sign_args += "-SignTool"
        $sign_args += $sign_tool
    }
    if ($sign_certificate_path) {
        $sign_args += "-CertificatePath"
        $sign_args += $sign_certificate_path
    }
    if ($sign_certificate_thumbprint) {
        $sign_args += "-CertificateThumbprint"
        $sign_args += $sign_certificate_thumbprint
    }
    $sign_args += "-Path"
    $sign_args += (Join-Path -Path $start_dir -ChildPath "dist\image2image\image2image.exe")
    & (Join-Path -Path $start_dir -ChildPath "sign.ps1") @sign_args
    echo "Signed application executable"
}

if ($zip) {
    echo "Zipping files..."
    python zip.py
    echo "Zipped files"
}

if ($installer) {
    echo "Building installer..."
    & (Join-Path -Path $start_dir -ChildPath "installer.ps1")
    echo "Built installer"
}

if ($sign -and $installer) {
    echo "Signing installer..."
    [string[]]$sign_args = @()
    if ($sign_tool) {
        $sign_args += "-SignTool"
        $sign_args += $sign_tool
    }
    if ($sign_certificate_path) {
        $sign_args += "-CertificatePath"
        $sign_args += $sign_certificate_path
    }
    if ($sign_certificate_thumbprint) {
        $sign_args += "-CertificateThumbprint"
        $sign_args += $sign_certificate_thumbprint
    }
    $installer_path = Get-ChildItem -Path (Join-Path -Path $start_dir -ChildPath "dist") -Filter "image2image-*-win_amd64-setup.exe" -File |
        Sort-Object -Property LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $installer_path) {
        throw "Could not find installer executable to sign."
    }
    $sign_args += "-Path"
    $sign_args += $installer_path.FullName
    & (Join-Path -Path $start_dir -ChildPath "sign.ps1") @sign_args
    echo "Signed installer"
}

if ($run) {
    cd dist
    Start-Process "run_image2image.bat"
    cd ../
}