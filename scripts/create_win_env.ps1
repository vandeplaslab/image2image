# Create Windows environment for development
uv venv venv2 -p 3.10

# Activate the virtual environment
.\venv2\Scripts\Activate.ps1

# Install the required packages
uv pip install PySide2
uv pip install -e .

# Install other packages
cd ../image2image-io
uv pip install -e ".[dev]"

cd ../image2image-reg
uv pip install -e .

cd ../qtextra
uv pip install -e .

cd ../qtreload
uv pip install -e .

cd ../koyo
uv pip install -e .

cd ../image2image
