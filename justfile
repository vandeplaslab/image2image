# ---- install ----

# install package in editable mode with dev dependencies
install:
    uv sync --group dev

# ---- lint & typecheck ----

# lint: check code with ruff (no auto-fix)
lint:
    ruff check .

# typecheck: run mypy type checking
typecheck:
    uvx mypy

# check-manifest: verify package manifest
check-manifest:
    uvx check-manifest

# ---- format ----

# fmt: format and auto-fix code with ruff
fmt:
    ruff format .
    ruff check . --fix --unsafe-fixes

# ---- test ----

# test: run the test suite (pass args, e.g. `just test tests/test_foo.py::test_name`)
test *args:
    uv run pytest --color=yes {{args}}

# test-cov: run tests with coverage (xml report for CI)
test-cov *args:
    uv run pytest --color=yes --cov=image2image --cov-report=xml {{args}}

# cov-html: run tests and generate an HTML coverage report
cov-html *args:
    uv run pytest --color=yes --cov=image2image --cov-report=html {{args}}
    @echo "→ open htmlcov/index.html"

# ---- CI / all checks ----

# check: run lint, typecheck, and tests (CI pipeline)
check: lint typecheck
    uv run pytest --color=yes

# ---- build ----

# dist: build source & wheel distributions
dist: check-manifest
    uv pip install -U build
    python -m build

# ---- docs ----

# docs-serve: serve documentation locally with live reload
docs-serve:
    uv run mkdocs serve

# docs-build: build static documentation site
docs-build:
    uv run mkdocs build

# ---- git ----

# pre: run pre-commit on all files
pre:
    prek run

# untrack: reset git index (use after updating .gitignore)
untrack:
    git rm -r --cached .
    git add .
    git commit -m ".gitignore fix"

# ---- utilities ----

# clean: remove build artifacts and cache directories
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    rm -rf dist build htmlcov .pytest_cache .mypy_cache .ruff_cache

# watch: re-run any target (e.g. `just watch test`)
watch *args:
    just {{args}}
