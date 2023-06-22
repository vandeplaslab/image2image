# Activate Rosetta
#arch -x86_64 /bin/zsh
#echo 'Activated Rosetta...'

# Create macOS environment
python3 -m virtualenv venv
source venv/bin/activate
echo 'Created venv...'

# Install dependencies
pip install -e ".[dev]"
echo 'Installed dependencies...'

# Install development libraries
cd ../qtreload
pip install -e .
echo 'Installed qtreload...'

cd ../qtextra
pip install -e .
echo 'Installed qtextra...'

cd ../napari-plot
pip install -e .
echo 'Installed napari-plot...'

# Install PyQt5
pip install PyQt5
