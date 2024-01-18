param (
    [switch]$update = $false,
    [switch]$update_app = $false,
    [switch]$debug = $false,
    [switch]$run = $false,
    [switch]$help = $false
)

if ($help) {
    Write-Output "Package app for Windows

    Output parameters:
    -update: Update all modules. Default=False
    -update_app: Update all modules. Default=False
    -debug: Add debugging statement. Default=False
    -no_docs: Don't build documentation. Default=False
    -run: Run application after it has been built.
    -help: Print this message.
    "
    Exit
}
# activate environment
conda activate image2image_package

$python_ver = &{python -V} 2>&1
echo "Python version "$python_ver

$start_dir = $pwd
echo "Current directory: " $start_dir.Path
$github_dir = $start_dir | Split-Path | Split-Path | Split-Path
echo "Github directory: " $github_dir

if ($update) {
    # Re-install qtextra
    echo "Re-installing qtextra..."
    $new_dir = Join-Path -Path $github_dir -ChildPath "qtextra" -Resolve
    cd $new_dir
    pip install .
    echo "Reinstalled qtextra"

    # Re-install koyo
    echo "Re-installing koyo..."
    $new_dir = Join-Path -Path $github_dir -ChildPath "koyo" -Resolve
    cd $new_dir
    pip install .
    echo "Reinstalled koyo"

    # Re-install image2image
    echo "Re-installing image2image-io..."
    $new_dir = Join-Path $github_dir -ChildPath "image2image-io" -Resolve
    cd $new_dir
    pip install -U .
    cd $start_dir
    echo "Reinstalled image2image-io"

    # Re-install image2image
    echo "Re-installing image2image..."
    $new_dir = Join-Path $github_dir -ChildPath "image2image" -Resolve
    cd $new_dir
    pip install -U .
    cd $start_dir
    echo "Reinstalled image2image"

    # Re-install napari (latest)
    echo "Re-installing napari..."
    pip install -U napari==0.4.18
    echo "Reinstalled napari"

    # Re-install PySide2
    echo "Re-installing PySide2..."
    pip install -U pyside2
    echo "Reinstalled PySide2"

    # Re-install pyinstaller
    echo "Re-installing pyinstaller..."
    pip install -U pyinstaller
    echo "Reinstalled pyinstaller"
}

# only update ionglow
if ($update_app) {
    # Re-install image2image
    echo "Re-installing image2image-io..."
    $new_dir = Join-Path $github_dir -ChildPath "image2image-io" -Resolve
    cd $new_dir
    pip install -U .
    cd $start_dir
    echo "Reinstalled image2image-io"

    echo "Re-installing image2image..."
    $new_dir = Join-Path -Path $github_dir -ChildPath "image2image" -Resolve
    cd $new_dir
    pip install -U .
    cd $start_dir
    echo "Reinstalled image2image"
}

# Get path
#$filename = "image2image.spec"
$filename = "image2image_split.spec"

# Build bundle
Write-Output "Debugging: $debug; Filename: $filename"
if ($debug) {
    pyinstaller.exe --noconfirm --clean --debug=all $filename
} else {
    pyinstaller.exe --noconfirm --clean $filename
}

conda deactivate image2image_package

# Copy runner script
Copy-Item -Path "run_image2image.bat" -Destination "dist/"

if ($run) {
    cd dist
    Start-Process "run_image2image.bat"
    cd ../
}