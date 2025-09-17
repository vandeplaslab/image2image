# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

image2image is a suite of Qt-based tools for image visualization and registration of whole-slide images and imaging mass spectrometry data. It's a scientific imaging application built with Python, Qt, and napari, designed for researchers working with microscopy and mass spectrometry data.

## Core Architecture

### Application Structure
- **Multi-tool architecture**: The application provides multiple specialized tools (launcher, viewer, register, crop, elastix, valis, convert, merge, fusion) accessible through a unified CLI interface
- **Qt-based GUI**: Built on QtPy for cross-platform compatibility with PySide2/6 and PyQt5/6 backends
- **Napari integration**: Uses napari for advanced image visualization and layer management
- **Plugin system**: Extensible through `image2image-io` (I/O operations) and `image2image-reg` (registration algorithms)

### Key Components
- **CLI interface** (`src/image2image/cli.py`): Main entry point with tool selection and argument parsing
- **Window base classes** (`src/image2image/qt/_dialog_base.py`): Shared functionality for all GUI applications
- **Viewer mixins** (`src/image2image/qt/_dialog_mixins.py`): Reusable components for image viewing and manipulation
- **Data models** (`src/image2image/models/`): Core data structures for handling imaging data
- **Readers system**: Extensible image format support through the `image2image-io` package

### Configuration & State Management
- Uses pydantic for configuration management
- Separate configs for different app components
- Settings stored in user directories following platform conventions
- Sentry integration for error reporting and telemetry

## Development Commands

### Installation & Setup
```bash
# Clone and install for development
git clone https://github.com/vandeplaslab/image2image.git
pip install -e .[dev]

# Install with specific Qt backend
pip install -e .[pyqt6]
pip install -e .[pyside6]
```

### Code Quality & Testing
```bash
# Run all pre-commit checks
make pre

# Type checking with mypy
make typecheck
tox -e mypy

# Run tests
pytest
pytest tests/

# Test with coverage
pytest --cov=image2image

# Check package manifest
make check-manifest
```

### Linting & Formatting
```bash
# Format code with ruff
ruff format .

# Lint and auto-fix issues  
ruff check . --fix

# Run specific linter checks
ruff check src/ tests/
```

### Building & Distribution
```bash
# Build distribution packages
make dist

# Build with uv (modern approach)
uv build

# Install build dependencies and build
pip install -U build
python -m build
```

### Running the Application
```bash
# Launch main application (launcher by default)
i2i
image2image

# Launch specific tools directly
i2viewer  # Image viewer
i2register  # Image registration
i2crop  # Image cropping
i2elastix  # Elastix registration
i2convert  # Format conversion

# With options
i2i --tool viewer --file path/to/image.tiff
i2viewer -f image1.tiff image2.czi --verbose
i2elastix --project_dir /path/to/project.i2reg
```

## Important Technical Details

### Supported Python Versions
- Minimum: Python 3.9
- Tested on: 3.9, 3.10, 3.11, 3.12, 3.13
- Dependencies locked to specific versions for stability (numpy<2, pandas<2, etc.)

### Qt Backend Considerations
- **QtPy abstraction layer**: Code should use `qtpy` imports, not direct Qt imports
- **Cross-platform support**: Windows, macOS, Linux all supported
- **Backend flexibility**: Supports PySide2/6 and PyQt5/6 through optional dependencies

### Development Environment
- **Modern Python tooling**: Uses `uv` for fast dependency management in CI
- **Strict code quality**: Comprehensive ruff configuration with many rules enabled
- **Type checking**: mypy with strict settings enabled
- **Pre-commit hooks**: Automated formatting, linting, and import organization

### Image Data Handling
- **Multi-format support**: .tiff, .czi, OME-TIFF and many others
- **Large data optimization**: Uses dask/xarray for memory-efficient processing
- **Multi-resolution support**: Pyramid/multi-scale image handling
- **Registration algorithms**: Supports both Elastix and VALIS registration methods

### Testing & CI
- **Multi-platform CI**: Tests on Ubuntu, Windows, macOS
- **Matrix testing**: Multiple Python versions and Qt backends
- **PyInstaller builds**: Automated creation of standalone executables
- **Coverage reporting**: CodeCov integration for test coverage

### Code Organization Patterns
- **Mixin architecture**: Shared functionality through mixins (`SingleViewerMixin`, `NoViewerMixin`)
- **Event-driven design**: Extensive use of Qt signals and slots
- **Configuration management**: Centralized config system with user-specific settings
- **Plugin architecture**: Modular design allowing optional registration and I/O packages

When working on this codebase, pay attention to the Qt-based architecture, the scientific imaging domain requirements, and the need for cross-platform compatibility. The application handles large image datasets, so memory efficiency and performance are important considerations.