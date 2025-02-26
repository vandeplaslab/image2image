param (
    [switch]$activate = $false,
    [switch]$update = $false,
    [switch]$update_app = $false,
    [switch]$update_deps = $false,
    [switch]$update_just_app = $false,
    [switch]$update_just_reader = $false,
    [switch]$update_just_register = $false,
    [switch]$update_pip = $false,
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
# activate environment
conda activate image2image_package
if ($activate) {
    Exit
}

if ($debug) {
    echo "Debug mode enabled"
    $env:PYINSTALLER_DEBUG="all"
} else {
    $env:PYINSTALLER_DEBUG="imports"
}

$python_ver = &{python -V} 2>&1
echo "Python version "$python_ver

$start_dir = $pwd
echo "Current directory: " $start_dir.Path
$github_dir = $start_dir | Split-Path | Split-Path | Split-Path
echo "Github directory: " $github_dir

[System.Collections.ArrayList]$local_install = @()
[System.Collections.ArrayList]$pip_install = @()

# update all dependencies and app
if ($update) {
    $update_deps = $true
    $update_app = $true
    $update_pip = $true
}

# update all dependencies and app
if ($update_pip) {
    $pip_install.Add("napari==0.5.6")
    $pip_install.Add("pydantic<2")
    $pip_install.Add("pyside2")
    $pip_install.Add("pyinstaller")
}

# only update app
if ($update_app) {
    $update_just_app = $true
    $update_just_reader = $true
    $update_just_register = $true
}

if ($update_just_app) {
    $local_install.Add("image2image")
}

if ($update_just_reader) {
    $local_install.Add("image2image-io")
}

if ($update_just_register) {
    $local_install.Add("image2image-reg")
}

# only update dependencies
$qtextra = $false
if ($update_deps) {
    $qtextra = $true
    $local_install.Add("koyo")
}

# install qtextra
if ($qtextra) {
    echo "Re-installing qtextra..."
    $new_dir = Join-Path -Path $github_dir -ChildPath "qtextra" -Resolve
    cd $new_dir
    pip install ".[sentry,console]"
    cd $start_dir
    echo "Reinstalled qtextra"

    echo "Re-installing qtextraplot..."
    $new_dir = Join-Path -Path $github_dir -ChildPath "qtextraplot" -Resolve
    cd $new_dir
    pip install ".[2d]"
    cd $start_dir
    echo "Reinstalled qtextraplot"
}

# install local packages
foreach ($package in $local_install) {
    echo "Re-installing $package..."
    $new_dir = Join-Path -Path $github_dir -ChildPath $package -Resolve
    cd $new_dir
    pip install .
    cd $start_dir
    echo "Reinstalled $package"
}

# install pip packages
foreach ($package in $pip_install) {
    echo "Re-installing $package..."
    pip install -U $package
    echo "Reinstalled $package"
}


# Get path
$filename = "image2image.spec"

# Build bundle
Write-Output "Filename: $filename"
# if ($debug) {
pyinstaller.exe --noconfirm --clean $filename --hide-console hide-early

# Copy runner script
Copy-Item -Path "run_image2image.bat" -Destination "dist/"

if ($zip) {
    echo "Zipping files..."
    python zip.py
    echo "Zipped files"
}

if ($run) {
    cd dist
    Start-Process "run_image2image.bat"
    cd ../
}
conda deactivate image2image_package