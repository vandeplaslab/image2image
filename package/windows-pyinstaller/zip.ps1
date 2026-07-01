# activate environment
$start_dir = $pwd
echo "Current directory: " $start_dir.Path
$github_dir = $start_dir | Split-Path | Split-Path | Split-Path
echo "Github directory: " $github_dir
$i2i_dir = Join-Path -Path $github_dir -ChildPath "image2image" -Resolve
# activate venv environment
$venv_activate = Join-Path -Path $i2i_dir -ChildPath "venv_package" | Join-Path -ChildPath "Scripts" | Join-Path -ChildPath "activate.ps1"
echo "Venv script location: " $venv_activate
& $venv_activate

# Report Python version
$python_ver = &{python -V} 2>&1
echo "Python version "$python_ver

# Zip files
python zip.py
echo "Zipped files"
