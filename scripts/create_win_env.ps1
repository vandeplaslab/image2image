# Create Windows environment for development
uv venv venv_package -p 3.10

# Activate the virtual environment
.\venv_package\Scripts\Activate.ps1

# Install the required packages
uv pip install pyqt6
uv pip install -e .

# Install other packages
cd ../image2image-io
uv pip install -e ".[dev]"

cd ../image2image-reg
uv pip install -e .

cd ../qtextra
uv pip install -e ".[console,sentry,pyqt6]"

cd ../qtextraplot
uv pip install -e ".[2d]"

cd ../qtreload
uv pip install -e .

cd ../koyo
uv pip install -e .

cd ../image2image
