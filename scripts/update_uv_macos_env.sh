# Activate Rosetta
#arch -x86_64 /bin/zsh
#echo 'Activated Rosetta...'

# Create macOS environment
#python3.9 -m virtualenv venv
#source venv/bin/activate
#echo 'Created venv...'

# Install PyQt5
#pip install PyQt5

# Install dependencies
uv pip install -e ".[pyqt6,dev]"
echo 'Installed dependencies...'

# Install development libraries
cd ../qtreload
uv pip install -e .
echo 'Installed qtreload...'

cd ../qtextra
uv pip install -e ".[sentry,console]"
echo 'Installed qtextra...'

cd ../qtextraplot
uv pip install -e ".[2d]"
echo 'Installed qtextraplot...'

cd ../koyo
uv pip install -e "."
echo 'Installed koyo...'
