"""Backend model discovery and explicit, atomic downloads into the application cache."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np

from image2image.models.annotation import FeatureSettings
from image2image.utils._appdirs import USER_CACHE_DIR
from image2image.utils.annotation.io import Cancel, check_cancel

TaskProgress = Callable[[str, int, int | None, str], None]


def read_classifier_metadata(path: str | Path) -> dict[str, Any]:
    """Read classifier settings without constructing its trees; support older artifacts."""
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
    FeatureSettings.model_validate(metadata["settings"])
    if (
        not isinstance(metadata.get("classes"), list)
        or not metadata["classes"]
        or not isinstance(metadata.get("name"), str)
        or not isinstance(metadata.get("sklearn"), str)
    ):
        raise ValueError("Invalid classifier metadata.")
    return {
        key: metadata[key]
        for key in ("name", "settings", "classes", "class_names", "sklearn", "samples_per_class", "trees")
        if key in metadata
    }


def model_cache_directory(backend: str) -> Path:
    """Return the managed directory for one optional inference backend."""
    if backend not in ("micro-sam", "cellpose"):
        raise ValueError(f"Unknown pretrained backend: {backend}")
    return USER_CACHE_DIR / "models" / backend


def pretrained_model_help(backend: str, name: str) -> str:
    """Explain registered model names using the backend authors' model documentation."""
    if not name:
        return "Select a model to see its purpose, encoder size, and requirements."
    if backend == "micro-sam":
        size = {"b": "Base", "l": "Large", "h": "Huge", "t": "Tiny"}.get(name.split("_")[1] if "_" in name else "", "")
        domain = next(
            (
                text
                for suffix, text in (
                    (
                        "_lm",
                        "Microscopy generalist for cells in light microscopy, including fluorescence and brightfield images.",
                    ),
                    ("_em_organelles", "Electron microscopy model for mitochondria and nuclei."),
                    ("_histopathology", "PathoSAM model for nuclei in histopathology images."),
                    ("_medical_imaging", "MedicoSAM generalist for medical image segmentation."),
                )
                if name.endswith(suffix)
            ),
            "Original Segment Anything model for general images, without microscopy fine-tuning.",
        )
        return f"{name}: {domain}\nEncoder: Vision Transformer {size or '(size not described)'}. Larger encoders use more memory; Tiny requires MobileSAM.\nLM means light microscopy; EM means electron microscopy."
    descriptions = {
        "cpsam_v2": "CellposeSAM v2, using the SAM Large encoder. Updated training reduces spurious masks in low-contrast regions.",
        "cpsam": "Original CellposeSAM model, using the SAM Large encoder, for general cell segmentation.",
        "cpdino": "CellposeDINO model, using the DINOv3 Large encoder, for cell segmentation. Requires dinov3.",
        "cpdino-vitb": "CellposeDINO with the smaller DINOv3 Base encoder. Requires dinov3.",
        "cyto": "Original Cellpose cytoplasm model for whole-cell segmentation.",
        "cyto2": "Cellpose 2 generalist cytoplasm model for whole-cell segmentation.",
        "cyto3": "Cellpose 3 generalist cytoplasm model for whole-cell segmentation.",
        "nuclei": "Cellpose model for nucleus segmentation.",
    }
    return f"{name}: {descriptions.get(name, 'Installed Cellpose registry entry; no additional model description is available.')}"


def cleanup_model_download(backend: str, download_id: str) -> None:
    """Remove only the unpublished files owned by one stopped download job."""
    if not download_id.isalnum():
        raise ValueError("Invalid download job ID.")
    for path in model_cache_directory(backend).glob(f".{download_id}.*.part"):
        path.unlink(missing_ok=True)


def _registry(backend: str) -> list[dict[str, Any]]:
    directory = model_cache_directory(backend)
    if backend == "micro-sam":
        os.environ["MICROSAM_CACHEDIR"] = str(directory)
        from micro_sam.util import VIT_T_SUPPORT, models

        registry = models()
        return [
            {
                "name": name,
                "reason": "Tiny SAM needs the optional mobile_sam architecture, which is missing or could not be imported. Downloading weights cannot supply it. Install MobileSAM in the app's Python environment, then restart."
                if name.startswith("vit_t") and not VIT_T_SUPPORT
                else "",
                "assets": [
                    {"name": key, "url": registry.urls[key], "hash": registry.registry[key]}
                    for key in (name, f"{name}_decoder")
                    if key in registry.registry
                ],
            }
            for name in registry.registry
            if not name.endswith("_decoder")
        ]
    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(directory)
    from cellpose import models as cellpose_models
    from cellpose.models import _MODEL_URL, MODEL_NAMES

    return [
        {
            "name": name,
            "reason": "CellposeDINO needs the optional dinov3 architecture, which is missing or could not be imported. Downloading weights cannot supply it. Install DINOv3 in the app's Python environment, then restart."
            if name.startswith("cpdino") and (find_spec("dinov3") is None or not hasattr(cellpose_models, "CPDINO"))
            else "",
            "assets": [{"name": name, "url": _MODEL_URL + name, "hash": ""}],
        }
        for name in MODEL_NAMES
    ]


def _receipt_path(directory: Path, name: str) -> Path:
    return directory / f".{name}.json"


def _cached(directory: Path, entry: dict[str, Any]) -> bool:
    try:
        receipt = json.loads(_receipt_path(directory, entry["name"]).read_text())
        for asset in entry["assets"]:
            stat = (directory / asset["name"]).stat()
            saved = receipt[asset["name"]]
            if stat.st_size != saved["size"] or stat.st_mtime_ns != saved["mtime_ns"]:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    else:
        return True


def list_pretrained_models(backend: str) -> list[dict[str, Any]]:
    """Discover installed registry entries and cache status without network access."""
    directory = model_cache_directory(backend)
    return [
        {**entry, "cached": _cached(directory, entry), "path": str(directory / entry["name"])}
        for entry in _registry(backend)
    ]


def _digest(path: Path, algorithm: str = "sha256", cancel: Cancel | None = None) -> str:
    if algorithm == "xxh128":
        from xxhash import xxh128

        digest = xxh128()
    else:
        digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            check_cancel(cancel)
            digest.update(block)
    return digest.hexdigest()


def download_pretrained_model(
    backend: str,
    name: str,
    *,
    cancel: Cancel | None = None,
    task_progress: TaskProgress | None = None,
    download_id: str | None = None,
) -> dict[str, Any]:
    """Download a selected model and decoder, verifying known hashes before publishing."""
    entries = _registry(backend)
    entry = next((entry for entry in entries if entry["name"] == name), None)
    if entry is None:
        raise ValueError(f"Unknown {backend} model: {name}")
    if entry["reason"]:
        raise ValueError(entry["reason"])
    directory = model_cache_directory(backend)
    directory.mkdir(parents=True, exist_ok=True)
    download_id = download_id or uuid4().hex
    if not download_id.isalnum():
        raise ValueError("Invalid download job ID.")
    receipt: dict[str, Any] = {}
    previously_cached = _cached(directory, entry)
    previous_receipt = json.loads(_receipt_path(directory, name).read_text()) if previously_cached else {}
    for asset in entry["assets"]:
        check_cancel(cancel)
        destination = directory / asset["name"]
        expected = asset["hash"]
        reusable = previously_cached
        if destination.is_file() and expected:
            algorithm, value = expected.split(":", 1)
            reusable = _digest(destination, algorithm, cancel) == value
        elif reusable:
            reusable = _digest(destination, cancel=cancel) == previous_receipt[asset["name"]].get("sha256")
        if not reusable:
            temporary = directory / f".{download_id}.{asset['name']}.part"
            try:
                if not asset["url"].startswith("https://"):
                    raise ValueError("Model downloads require HTTPS.")
                request = Request(asset["url"], headers={"User-Agent": "image2image model downloader"})  # noqa: S310 - HTTPS registry URL checked above
                with urlopen(request, timeout=30) as response, temporary.open("wb") as stream:  # noqa: S310 - URLs come from installed backend registries
                    total = int(response.headers.get("Content-Length", 0)) or None
                    completed = 0
                    read = getattr(response, "read1", response.read)
                    while block := read(256 * 1024):
                        check_cancel(cancel)
                        stream.write(block)
                        completed += len(block)
                        if task_progress:
                            task_progress(asset["name"], completed, total, f"Downloading {asset['name']}")
                check_cancel(cancel)
                if not completed or (total is not None and completed != total):
                    raise OSError(f"Incomplete model download: {asset['name']}")
                if expected:
                    algorithm, value = expected.split(":", 1)
                    if _digest(temporary, algorithm, cancel) != value:
                        raise ValueError(f"Model checksum does not match: {asset['name']}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        stat = destination.stat()
        receipt[asset["name"]] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _digest(destination, cancel=cancel),
        }
    check_cancel(cancel)
    target = _receipt_path(directory, name)
    temporary = target.with_name(f".{download_id}.receipt.part")
    try:
        temporary.write_text(json.dumps(receipt))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {**entry, "cached": True, "path": str(directory / name)}


def resolve_pretrained_model(backend: str, name: str, *, allow_download: bool = False) -> tuple[str, str]:
    """Resolve complete managed weights, optionally downloading for explicit API callers."""
    entries = list_pretrained_models(backend)
    entry = next((entry for entry in entries if entry["name"] == name), None)
    if entry is None:
        raise ValueError(f"Unknown {backend} model: {name}")
    if entry["reason"]:
        raise ValueError(entry["reason"])
    if not entry["cached"]:
        if not allow_download:
            raise ValueError(f"Download {name} in Predict or choose a local checkpoint first.")
        entry = download_pretrained_model(backend, name)
    decoder = next((asset["name"] for asset in entry["assets"] if asset["name"].endswith("_decoder")), "")
    directory = model_cache_directory(backend)
    return str(directory / name), str(directory / decoder) if decoder else ""
