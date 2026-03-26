# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`image2image` is a suite of Qt-based tools for image visualization and registration of whole-slide images and imaging mass spectrometry data. It is built on Python, QtPy (cross-platform Qt abstraction), and napari.

## Development Commands

```bash
# Install for development
pip install -e .[dev]

# Install with specific Qt backend
pip install -e .[pyqt6]
pip install -e .[pyside6]

# Run tests
pytest
pytest tests/test_viewer.py          # single test file
pytest tests/test_viewer.py::test_fn # single test

# Lint and format
ruff format .
ruff check . --fix

# Type checking
make typecheck                        # runs mypy
tox -e mypy

# Pre-commit (runs ruff, pycln, black, absolufy-imports)
make pre

# Build distribution
make dist
uv build
```

## Architecture

### Multi-tool structure

The app exposes multiple specialized tools through a unified CLI (`src/image2image/cli.py`):

| Entry point | Tool |
|---|---|
| `i2i` / `image2image` | Main launcher |
| `i2viewer` | Image viewer |
| `i2register` | Image registration |
| `i2elastix` | Elastix-based registration |
| `i2crop` | Cropping utility |
| `i2convert` | Format conversion |

### Key layers

- **`qt/_dialog_base.py`** — `Window` base class (~878 lines). All tool dialogs inherit from this. Provides napari viewer integration, Qt signal wiring, version checking, and drag-and-drop support.
- **`qt/_dialog_mixins.py`** — `SingleViewerMixin` and `NoViewerMixin` for dialogs that need one or zero napari viewers. Compose with the base class.
- **`qt/dialog_*.py`** — Individual tool dialogs (viewer, register, elastix, valis, crop, convert, merge, fusion, elastix3d). Each composes the base class + mixins.
- **`qt/launcher.py`** — Tile-based launcher GUI that opens individual tool dialogs.
- **`models/`** — Pydantic V2 data models: `DataModel` (image data container), `Transformation`, `TransformData`, WSI preprocessing models.
- **`config.py`** — Pydantic-based configuration. Separate config classes per tool (ViewerConfig, RegisterConfig, etc.), stored in user platform directories.
- **`utils/`** — Standalone utilities: cropping, transforms, fiducials, VALIS integration, download helpers.

### External packages

The core I/O and registration logic lives in separate packages:
- **`image2image-io`** (>=0.3.0) — multi-format image reading (TIFF, CZI, OME-TIFF, etc.), `ImageWrapper`, `BaseReader`
- **`image2image-reg`** (>=0.3.0) — registration algorithms (Elastix, VALIS)

Changes to image format support or registration algorithms should be made in those packages, not here.

### Qt conventions

- Always import from `qtpy`, not directly from `PyQt5/6` or `PySide2/6`.
- Use Qt signals/slots for cross-widget communication, not direct method calls between dialogs.
- Supported backends: PySide2, PySide6, PyQt5, PyQt6 (all via optional deps).

### Code quality settings

- **Line length:** 120 characters
- **Formatter:** ruff (not black directly, though black is in pre-commit for legacy compatibility)
- **Import style:** absolute imports enforced by `absolufy-imports` pre-commit hook
- **Ruff rules:** very broad set enabled (F, E, W, UP, I, S, D, B, SIM, RET, TRY, RUF, and more) — run `ruff check . --fix` before committing
- **mypy:** strict mode enabled