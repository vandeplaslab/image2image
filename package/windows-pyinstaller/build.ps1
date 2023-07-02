param (
    [switch]$update = $false,
    [switch]$update_i2i = $false,
    [switch]$debug = $false,
    [switch]$run = $false,
    [switch]$help = $false
)

if ($help) {
    Write-Output "Package app for Windows

    Output parameters:
    -update: Update all modules. Default=False
    -update_i2i: Update all modules. Default=False
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

if ($update) {
    # Re-install pyinstaller
    echo "Re-installing pyinstaller..."
    pip install -U pyinstaller
    echo "Reinstalled pyinstaller"

    # Re-install qtextra
    echo "Re-installing qtextra..."
    cd ..\..\..\qtextra
    pip install .
    cd ..\
    echo "Reinstalled qtextra"

    # Re-install napari (latest)
    echo "Re-installing napari..."
    pip install -U napari==0.4.17
    echo "Reinstalled napari"

    # Re-install PySide2
    echo "Re-installing PySide2..."
    pip install -U pyside2
    echo "Reinstalled PySide2"

    # Re-install image2image
    echo "Re-installing image2image..."
    cd ../image2image
    pip install -U .
    cd package/windows-pyinstaller
    echo "Reinstalled image2image"
}

# only update ionglow
if ($update_i2i) {
    echo "Re-installing image2image..."
    cd ../../
    pip install -U .
    cd package/windows-pyinstaller
    echo "Reinstalled image2image"
}

# Get path
$filename = "image2image.spec"

# Build bundle
Write-Output "Debugging: $debug; Filename: $filename"
if ($debug) {
    pyinstaller.exe --onedir --noconfirm --clean --debug=all $filename
} else {
    pyinstaller.exe --onedir --noconfirm --clean $filename
}

conda deactivate imimsui_package

# Copy runner script
Copy-Item -Path "run_image2image.bat" -Destination "dist/"

if ($run) {
    cd dist
    Start-Process "run_image2image.bat"
    cd ../
}