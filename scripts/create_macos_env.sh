# Create macOS environment
python3 -m virtualenv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Install other libraries
cd ../qtreload
pip install -e .
cd ../qtextra
pip install -e .

