# Install the required packages
uv pip install -e ".[pyqt6,dev]"

# Install other packages
cd ../image2image-io
uv pip install -e .

cd ../image2image-reg
uv pip install -e .

cd ../qtextra
uv pip install -e ".[sentry,console]"

cd ../qtextraplot
uv pip install -e ".[2d]"

cd ../qtreload
uv pip install -e .

cd ../koyo
uv pip install -e .

cd ../image2image
