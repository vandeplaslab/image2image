"""Bounded random-forest training and optional local pretrained inference."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import numpy as np
from shapely.geometry import box, shape
from shapely.ops import unary_union

from image2image.models.annotation import AnnotationObject, FeatureSettings, PredictionBatch, PredictionRequest
from image2image.utils.annotation.images import PatchReader, fingerprint
from image2image.utils.annotation.io import Cancel, Progress, _progress, check_cancel
from image2image.utils.annotation.model_cache import TaskProgress, resolve_pretrained_model
from image2image.utils.annotation.store import AnnotationStore


def features(image: np.ndarray, settings: FeatureSettings) -> np.ndarray:
    """Compute deterministic intensity, edge, and texture features from HWC input."""
    from skimage.feature import multiscale_basic_features

    data = image.astype(np.float32)
    # Fixed scaling prevents patch-dependent normalization seams.
    if np.issubdtype(image.dtype, np.integer):
        data /= np.iinfo(image.dtype).max
    return np.asarray(
        multiscale_basic_features(  # type: ignore[no-untyped-call]
            data,
            intensity=True,
            edges=True,
            texture=True,
            sigma_min=settings.sigma_min,
            sigma_max=settings.sigma_max,
            channel_axis=-1,
        ),
        dtype=np.float32,
    )


def _save_forest(path: Path, forest: Any, metadata: dict) -> None:
    arrays: dict[str, Any] = {
        "metadata": np.asarray(json.dumps(metadata)),
        "n_features": np.asarray(forest.n_features_in_),
    }
    for index, estimator in enumerate(forest.estimators_):
        state = estimator.tree_.__getstate__()
        arrays[f"nodes_{index}"] = state["nodes"]
        arrays[f"values_{index}"] = state["values"]
        arrays[f"depth_{index}"] = np.asarray(state["max_depth"])
    np.savez_compressed(path, **arrays)


def load_forest(path: str | Path) -> tuple[Any, dict]:
    """Load this app's non-pickle forest artifact; reject incompatible sklearn versions."""
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, _tree

    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        if metadata["sklearn"].split(".")[:2] != sklearn.__version__.split(".")[:2]:
            raise ValueError("Classifier sklearn version differs. Retrain using the saved training annotations.")
        n_features, n_classes = int(arrays["n_features"]), len(metadata["classes"])
        forest = RandomForestClassifier(
            n_estimators=metadata["trees"],
            n_jobs=1,
            max_depth=metadata["settings"].get("max_depth", 24),
            random_state=metadata["settings"].get("seed", 42),
        )
        forest.n_features_in_, forest.n_classes_, forest.n_outputs_ = n_features, n_classes, 1
        forest.classes_ = np.arange(n_classes)
        forest.estimators_ = []
        for index in range(metadata["trees"]):
            estimator = DecisionTreeClassifier()
            estimator.n_features_in_, estimator.n_classes_, estimator.n_outputs_ = n_features, n_classes, 1
            estimator.classes_ = np.arange(n_classes)
            estimator.tree_ = _tree.Tree(n_features, np.array([n_classes], dtype=np.intp), 1)
            nodes = arrays[f"nodes_{index}"]
            estimator.tree_.__setstate__(
                {
                    "max_depth": int(arrays[f"depth_{index}"]),
                    "node_count": len(nodes),
                    "nodes": nodes,
                    "values": arrays[f"values_{index}"],
                }
            )
            forest.estimators_.append(estimator)
    return forest, metadata


def train_classifier(
    store: AnnotationStore,
    name: str,
    settings: FeatureSettings,
    *,
    cancel: Cancel | None = None,
    progress: Progress | None = None,
) -> Path:
    """Train on included accepted examples, masking same-set Ignore regions before sampling."""
    import sklearn
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from sklearn.ensemble import RandomForestClassifier

    if settings.sigma_max < settings.sigma_min:
        raise ValueError("Maximum feature scale must be at least the minimum scale.")
    revision = store.revision
    input_slides = {slide.id: slide for slide in store.slides()}
    classes = store.classes()
    trainable = {item.id for item in classes if not item.is_ignore}
    if settings.class_ids is not None:
        if set(settings.class_ids) - {item.id for item in classes}:
            raise ValueError("Some selected training classes no longer exist.")
        trainable.intersection_update(settings.class_ids)
    ignore_ids = {item.id for item in classes if item.is_ignore}
    class_ids = sorted(
        {obj.class_id for obj in store.iter_objects(training=True) if obj.class_id and obj.class_id in trainable}
    )
    if len(class_ids) < 2:
        raise ValueError(
            "Include accepted annotations for at least two classes, including background where appropriate."
        )
    rng = np.random.default_rng(settings.seed)
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    training: list[dict[str, Any]] = []
    ignore_regions: dict[int, dict[str, Any]] = {}
    tile_size = 256
    ds = settings.downsample
    halo = math.ceil(4 * settings.sigma_max)
    for slide in store.slides():
        examples = []
        for obj in store.iter_objects(slide_id=slide.id, training=True):
            check_cancel(cancel)
            if obj.class_id not in trainable:
                continue
            masks = []
            for class_id in ignore_ids:
                for mask in store.iter_objects(
                    slide_id=slide.id, set_id=obj.set_id, class_id=class_id, bounds=obj.geometry.bounds
                ):
                    masks.append(mask.geometry)
                    ignore_regions[mask.key] = {
                        "slide_id": slide.id,
                        "set_id": mask.set_id,
                        "key": mask.key,
                        "class_id": mask.class_id,
                        "geometry": mask.geometry.wkb_hex,
                    }
            geometry = obj.geometry.difference(unary_union(masks)) if masks else obj.geometry
            if not geometry.is_empty and geometry.area > 0:
                examples.append(replace(obj, geometry=geometry))
        if not examples:
            continue
        if fingerprint(store.resolve_path(slide)) != slide.fingerprint:
            raise ValueError("A training image changed. Reload its metadata before training.")
        for index, obj in enumerate(examples):
            if obj.class_id is None:
                continue
            for other in examples[index + 1 :]:
                if (
                    other.class_id
                    and other.class_id != obj.class_id
                    and obj.geometry.intersection(other.geometry).area > 1e-9
                ):
                    raise ValueError(
                        "Training examples from different classes overlap. Correct the conflicting examples."
                    )
        training.extend(
            {
                "slide_id": slide.id,
                "set_id": obj.set_id,
                "key": obj.key,
                "id": obj.object_id,
                "class_id": obj.class_id,
                "geometry": obj.geometry.wkb_hex,
            }
            for obj in examples
        )
        with PatchReader(store.resolve_path(slide), series=slide.series) as reader:
            # A disk-backed tile set keeps even very large marked training regions bounded.
            import sqlite3

            with TemporaryDirectory(dir=store.path / "staging") as directory:
                tiles = sqlite3.connect(Path(directory) / "tiles.sqlite")
                try:
                    tiles.execute("CREATE TABLE tiles(x INTEGER,y INTEGER,PRIMARY KEY(x,y))")
                    for obj in examples:
                        if not obj.class_id:
                            continue
                        x0, y0, x1, y1 = obj.geometry.bounds
                        for y in range(max(0, int(y0) // (tile_size * ds)), math.ceil(y1 / (tile_size * ds))):
                            for x in range(max(0, int(x0) // (tile_size * ds)), math.ceil(x1 / (tile_size * ds))):
                                tiles.execute("INSERT OR IGNORE INTO tiles VALUES(?,?)", (x, y))
                    tiles.commit()
                    for index, (tx, ty) in enumerate(tiles.execute("SELECT x,y FROM tiles ORDER BY y,x")):
                        check_cancel(cancel)
                        x0, y0 = tx * tile_size * ds, ty * tile_size * ds
                        x1, y1 = min(x0 + tile_size * ds, reader.shape[1]), min(y0 + tile_size * ds, reader.shape[0])
                        patch = reader.read(
                            (x0 - halo * ds, y0 - halo * ds, x1 + halo * ds, y1 + halo * ds),
                            downsample=ds,
                            channels=settings.channels,
                        )
                        transform = Affine(
                            patch.downsample[0], 0, patch.offset[0], 0, patch.downsample[1], patch.offset[1]
                        )
                        labels = rasterize(
                            [(obj.geometry, class_ids.index(obj.class_id) + 1) for obj in examples if obj.class_id],
                            out_shape=patch.data.shape[:2],
                            transform=transform,
                            dtype=np.int32,
                        )
                        core = rasterize(
                            [(box(x0, y0, x1, y1), 1)], out_shape=labels.shape, transform=transform, dtype=np.uint8
                        )
                        matrix = features(patch.data, settings)
                        for class_index, class_id in enumerate(class_ids, 1):
                            values = matrix[(labels == class_index) & (core != 0)]
                            if not len(values):
                                continue
                            priorities = rng.random(len(values))
                            if class_id in samples:
                                old_values, old_priorities = samples[class_id]
                                values = np.concatenate((old_values, values))
                                priorities = np.concatenate((old_priorities, priorities))
                            keep = np.argsort(priorities)[-settings.samples_per_class :]
                            samples[class_id] = values[keep], priorities[keep]
                        _progress(progress, index, f"Sampling training examples from {slide.name}")
                finally:
                    tiles.close()
    if set(samples) != set(class_ids):
        raise ValueError(
            "Some included classes have no usable pixels after Ignore masking at this resolution. "
            "Check Ignore regions, include more examples, or use a smaller downsample."
        )
    per_class = min(len(samples[class_id][0]) for class_id in class_ids)
    train_x = np.concatenate([samples[class_id][0][:per_class] for class_id in class_ids])
    train_y = np.repeat(np.arange(len(class_ids)), per_class)
    check_cancel(cancel)
    forest = RandomForestClassifier(
        n_estimators=settings.trees, random_state=settings.seed, n_jobs=1, max_depth=settings.max_depth
    )
    _progress(progress, 0, f"Training {settings.trees} trees on {len(train_y):,} balanced samples")
    forest.fit(train_x, train_y)
    check_cancel(cancel)
    if store.revision != revision:
        raise ValueError("Training inputs changed. Train again with the current annotations.")
    for slide_id in {example["slide_id"] for example in training}:
        slide = input_slides[slide_id]
        if fingerprint(store.resolve_path(slide)) != slide.fingerprint:
            raise ValueError("A training image changed during training. Train again with the current image.")
    metadata = {
        "schema_version": 1,
        "name": name,
        "settings": settings.model_dump(),
        "classes": class_ids,
        "class_names": {item.id: item.name for item in store.classes()},
        "trees": settings.trees,
        "sklearn": sklearn.__version__,
        "revision": revision,
        "samples_per_class": per_class,
        "training": training,
        "ignore_regions": list(ignore_regions.values()),
        "slides": [slide.model_dump() for slide in input_slides.values()],
        "numpy": np.__version__,
        "scikit_image": version("scikit-image"),
    }
    filename = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "classifier"
    path = store.path / "models" / f"{filename}-{uuid4().hex[:8]}.npz"
    temporary = path.with_name(f".{path.name}.tmp.npz")
    _save_forest(temporary, forest, metadata)
    temporary.replace(path)
    return path


def backend_available(name: str) -> bool:
    """Check optional package availability without importing torch or model packages."""
    from importlib.util import find_spec

    return name == "forest" or find_spec({"micro-sam": "micro_sam", "cellpose": "cellpose"}[name]) is not None


def _device(name: str) -> Any:
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select CPU or Auto.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Select CPU or Auto.")
    return torch.device(name)


def _pretrained(backend: str, model: str, checkpoint: str, device: str, allow_download: bool) -> Any:
    if not backend_available(backend):
        extra = "annotate-sam" if backend == "micro-sam" else "annotate-cellpose"
        raise ImportError(f"Install image2image[{extra}] in this app's Python environment.")
    decoder = ""
    if not checkpoint:
        checkpoint, decoder = resolve_pretrained_model(
            backend, model or ("vit_b_lm" if backend == "micro-sam" else "cpsam"), allow_download=allow_download
        )
    return _load_pretrained(
        backend,
        model,
        checkpoint,
        device,
        allow_download,
        fingerprint(Path(checkpoint)) if checkpoint and Path(checkpoint).is_file() else "",
        decoder,
        fingerprint(Path(decoder)) if decoder else "",
    )


@lru_cache(maxsize=2)
def _load_pretrained(
    backend: str,
    model: str,
    checkpoint: str,
    device: str,
    allow_download: bool,
    checkpoint_revision: str,
    decoder: str = "",
    decoder_revision: str = "",
) -> Any:
    if not backend_available(backend):
        extra = "annotate-sam" if backend == "micro-sam" else "annotate-cellpose"
        raise ImportError(f"Install image2image[{extra}] in this app's Python environment, then restart Annotate.")
    if checkpoint and not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    device_ = _device(device)
    if backend == "micro-sam":
        return _sam_bundle(model, checkpoint, decoder, device, checkpoint_revision, decoder_revision)[0]
    from cellpose.models import MODEL_NAMES, CellposeModel

    if not checkpoint and model and model not in MODEL_NAMES:
        raise ValueError(f"Unknown Cellpose model '{model}'. Select a supported model or a local checkpoint.")

    return CellposeModel(gpu=device_.type != "cpu", device=device_, pretrained_model=checkpoint or model or "cpsam")


@lru_cache(maxsize=2)
def _sam_bundle(
    model: str, checkpoint: str, decoder: str, device: str, checkpoint_revision: str, decoder_revision: str
) -> tuple[Any, dict[str, Any]]:
    from micro_sam.util import get_sam_model

    predictor, state = get_sam_model(
        model_type=model or "vit_b_lm",
        checkpoint_path=checkpoint,
        decoder_path=decoder or None,
        device=_device(device),
        return_state=True,
    )
    predictor._i2i_decoder_path = decoder
    # The live model already holds encoder weights; retain only the optional instance decoder state.
    return predictor, {"decoder_state": state["decoder_state"]} if "decoder_state" in state else {}


def _automatic_sam(request: PredictionRequest) -> tuple[Any, Any]:
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter

    checkpoint, decoder = request.checkpoint, ""
    if not checkpoint:
        checkpoint, decoder = resolve_pretrained_model(
            "micro-sam", request.model or "vit_b_lm", allow_download=request.allow_download
        )
    predictor, state = _sam_bundle(
        request.model,
        checkpoint,
        decoder,
        request.device,
        fingerprint(Path(checkpoint)),
        fingerprint(Path(decoder)) if decoder else "",
    )
    mode = "ais" if "decoder_state" in state else "amg"
    _predictor, segmenter = get_predictor_and_segmenter(
        request.model or "vit_b_lm",
        checkpoint=checkpoint,
        device=_device(request.device),
        predictor=predictor,
        state=state,
        segmentation_mode=mode,
        is_tiled=False,
    )
    predictor._i2i_segmentation_mode = mode
    return predictor, segmenter


def _sam_instance_masks(segmenter: Any, image: np.ndarray) -> np.ndarray:
    global _SAM_IMAGE_KEY

    # Automatic segmentation changes predictor embeddings; prompted runs must set their image again.
    _SAM_IMAGE_KEY = ""
    segmenter.initialize(image, verbose=False)
    return np.asarray(segmenter.generate(output_mode="instance_segmentation"))


_SAM_IMAGE_KEY: str = ""


def _rgb(data: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    if data.shape[-1] > 3:
        raise ValueError("Choose at most three channels for pretrained segmentation.")
    normalized = np.clip((data.astype(np.float32) - ranges[0]) / np.maximum(ranges[1] - ranges[0], 1e-12), 0, 1)
    if data.shape[-1] == 1:
        normalized = np.repeat(normalized, 3, axis=-1)
    elif data.shape[-1] == 2:
        normalized = np.concatenate((normalized, np.zeros_like(normalized[..., :1])), axis=-1)
    return np.asarray(normalized * 255, dtype=np.uint8)


@lru_cache(maxsize=8)
def _checkpoint_hash(path: str, modified: int, size: int) -> str:
    # Size and modification time form part of the cache key; file content is hashed once.
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _model_provenance(model: Any, backend: str, path: str = "") -> dict:
    path = path or str(getattr(model, "checkpoint_path", getattr(model, "pretrained_model", "")))
    try:
        package_version = version({"forest": "scikit-learn", "micro-sam": "micro-sam", "cellpose": "cellpose"}[backend])
    except PackageNotFoundError:
        package_version = "unavailable"
    result = {"package": backend, "version": package_version}
    if path and Path(path).is_file():
        stat = Path(path).stat()
        result.update(
            checkpoint=path,
            sha256=_checkpoint_hash(path, stat.st_mtime_ns, stat.st_size),
            fingerprint=fingerprint(Path(path)),
        )
    decoder = getattr(model, "_i2i_decoder_path", "")
    if decoder:
        stat = Path(decoder).stat()
        result["decoder"] = {
            "checkpoint": decoder,
            "fingerprint": fingerprint(Path(decoder)),
            "sha256": _checkpoint_hash(decoder, stat.st_mtime_ns, stat.st_size),
        }
    if hasattr(model, "_i2i_segmentation_mode"):
        result["segmentation_mode"] = model._i2i_segmentation_mode
    return result


def _prompt_mask(predictor: Any, rgb: np.ndarray, request: PredictionRequest, patch: Any) -> np.ndarray:
    global _SAM_IMAGE_KEY

    cache_key = hashlib.sha256(rgb.tobytes()).hexdigest() + str(id(predictor))
    if cache_key != _SAM_IMAGE_KEY:
        predictor.set_image(rgb)
        _SAM_IMAGE_KEY = cache_key
    points = np.asarray([(x, y) for x, y, _ in request.points], dtype=float).reshape(-1, 2)
    labels = np.asarray([value for _, _, value in request.points], dtype=np.int32)
    if len(points):
        points = (points - patch.offset) / patch.downsample
    prompt_box = None
    if request.box:
        prompt_box = ((np.asarray(request.box).reshape(2, 2) - patch.offset) / patch.downsample).reshape(4)
    masks, scores, _ = predictor.predict(
        point_coords=points if len(points) else None,
        point_labels=labels if len(labels) else None,
        box=prompt_box,
        multimask_output=True,
    )
    return np.asarray(masks[int(np.argmax(scores))], dtype=np.uint32)


def predict(
    store: AnnotationStore,
    request: PredictionRequest,
    *,
    cancel: Cancel | None = None,
    progress: Progress | None = None,
    task_progress: TaskProgress | None = None,
) -> PredictionBatch:
    """Stage complete tiled predictions for explicit review; never overwrite accepted objects."""
    import zarr
    from rasterio.features import shapes
    from rasterio.transform import Affine

    def report(phase: str, completed: int = 0, total: int | None = None) -> None:
        message = f"{phase}: {slide.name}"
        if task_progress:
            task_progress(phase, completed, total, message)
        _progress(progress, completed, message)

    if store.revision != request.revision:
        raise ValueError("Prediction inputs are stale. Queue the current inputs again.")
    slide = store.slide(request.slide_id)
    report("Preparing image")
    source_path = store.resolve_path(slide)
    if fingerprint(source_path) != slide.fingerprint:
        raise ValueError("Source image changed; reload its metadata before prediction.")
    provenance = {
        "request": request.model_dump(),
        "source_fingerprint": slide.fingerprint,
        "slide_geometry": slide.geometry.model_dump(),
    }
    set_id = store.create_set(slide.id, f"{request.backend} predictions", staging=True, metadata=provenance)
    count, ambiguous = 0, 0
    try:
        with PatchReader(source_path, series=slide.series) as reader:
            height, width = reader.shape
            bounds = request.bounds or (0, 0, width, height)
            x0, y0, x1, y1 = max(0, bounds[0]), max(0, bounds[1]), min(width, bounds[2]), min(height, bounds[3])
            if x1 <= x0 or y1 <= y0:
                raise ValueError("Prediction ROI does not intersect the slide.")  # noqa: TRY301 - the outer handler cleans up the unpublished set
            if request.backend == "micro-sam" and request.sam_mode == "prompts":
                negative_points = [point for point in request.points if point[2] == 0]
                prompts = [
                    request.model_copy(update={"points": [point, *negative_points], "box": None, "boxes": []})
                    for point in request.points
                    if point[2] == 1
                ]
                prompts.extend(
                    request.model_copy(update={"points": negative_points, "box": bounds_, "boxes": []})
                    for bounds_ in ([request.box] if request.box else []) + request.boxes
                )
                if not prompts:
                    raise ValueError("Add at least one foreground point or box; negative points only exclude pixels.")  # noqa: TRY301 - the outer handler cleans up the unpublished set
                patch = reader.read(
                    (x0, y0, x1, y1),
                    downsample=request.downsample,
                    channels=request.channels,
                    max_pixels=4_194_304 * len(request.channels),
                )
                ranges = np.percentile(patch.data, (1, 99), axis=(0, 1))
                report("Loading model")
                predictor = _pretrained(
                    request.backend, request.model, request.checkpoint, request.device, request.allow_download
                )
                predictor._i2i_segmentation_mode = "prompts"
                provenance.update(
                    model=_model_provenance(predictor, request.backend),
                    preprocessing={"percentile_ranges": ranges.tolist(), "output": "RGB uint8"},
                )
                rgb = _rgb(patch.data, ranges)
                report("Prompt inference", 0, len(prompts))
                for index, prompt in enumerate(prompts, 1):
                    check_cancel(cancel)
                    mask = _prompt_mask(predictor, rgb, prompt, patch)
                    check_cancel(cancel)
                    polygons = [shape(g) for g, v in shapes(mask.astype(np.int32), mask=mask != 0)]
                    if polygons:
                        geometry = patch.coordinates.transform(unary_union(polygons), slide.geometry)
                        store.append_staged(
                            set_id,
                            [
                                AnnotationObject(
                                    set_id,
                                    f"{request.id}-{index}",
                                    geometry,
                                    request.class_id,
                                    provenance={**provenance, "prompt": {"points": prompt.points, "box": prompt.box}},
                                    review="proposed",
                                )
                            ],
                        )
                        count += 1
                    report("Prompt inference", index, len(prompts))
            elif request.backend == "forest":
                report("Loading classifier")
                forest, metadata = load_forest(request.model)
                provenance.update(
                    model=_model_provenance(forest, request.backend, request.model), preprocessing=metadata["settings"]
                )
                settings = FeatureSettings.model_validate(metadata["settings"])
                if tuple(request.channels) != settings.channels or request.downsample != settings.downsample:
                    raise ValueError("Use the classifier's saved channels and downsample for prediction.")  # noqa: TRY301 - the outer handler cleans up the unpublished set
                if not set(metadata["classes"]).issubset({item.id for item in store.classes()}):
                    raise ValueError("This classifier references classes that are missing from the project.")  # noqa: TRY301 - the outer handler cleans up the unpublished set
                halo = max(request.halo, math.ceil(4 * settings.sigma_max))
                ds, tile_size = request.downsample, request.tile_size
                out_height, out_width = math.ceil((y1 - y0) / ds), math.ceil((x1 - x0) / ds)
                with TemporaryDirectory(dir=store.path / "staging") as directory:
                    root = zarr.open_group(directory, mode="w", zarr_format=2)
                    sums = root.create_array(
                        "sums",
                        shape=(out_height, out_width, len(metadata["classes"])),
                        chunks=(tile_size, tile_size, len(metadata["classes"])),
                        dtype="f4",
                        fill_value=0,
                    )
                    weights = root.create_array(
                        "weights",
                        shape=(out_height, out_width),
                        chunks=(tile_size, tile_size),
                        dtype="f4",
                        fill_value=0,
                    )
                    # Probability blending uses overlap; feature halos are excluded from blending.
                    stride = max(32, tile_size - 2 * min(halo, tile_size // 4))
                    total_tiles = math.ceil(out_height / stride) * math.ceil(out_width / stride)
                    completed_tiles = 0
                    report("Predicting patches", 0, total_tiles)
                    for ty in range(0, out_height, stride):
                        for tx in range(0, out_width, stride):
                            check_cancel(cancel)
                            tw, th = min(tile_size, out_width - tx), min(tile_size, out_height - ty)
                            px0, py0 = x0 + tx * ds, y0 + ty * ds
                            patch = reader.read(
                                (
                                    px0 - halo * ds,
                                    py0 - halo * ds,
                                    min(x1, px0 + tw * ds) + halo * ds,
                                    min(y1, py0 + th * ds) + halo * ds,
                                ),
                                downsample=ds,
                                channels=settings.channels,
                            )
                            matrix = features(patch.data, settings)
                            probabilities = forest.predict_proba(matrix.reshape(-1, matrix.shape[-1])).reshape(
                                *matrix.shape[:2], -1
                            )
                            from scipy.ndimage import map_coordinates

                            yy = (y0 + (ty + np.arange(th) + 0.5) * ds - patch.offset[1]) / patch.downsample[1] - 0.5
                            xx = (x0 + (tx + np.arange(tw) + 0.5) * ds - patch.offset[0]) / patch.downsample[0] - 0.5
                            gy, gx = np.meshgrid(yy, xx, indexing="ij")
                            core = np.stack(
                                [
                                    map_coordinates(probabilities[..., c], (gy, gx), order=1, mode="nearest")
                                    for c in range(probabilities.shape[-1])
                                ],
                                axis=-1,
                            )
                            blend = np.maximum(
                                0.01, np.outer(np.hanning(th + 2)[1:-1], np.hanning(tw + 2)[1:-1])
                            ).astype(np.float32)
                            sums[ty : ty + th, tx : tx + tw] = (
                                np.asarray(sums[ty : ty + th, tx : tx + tw]) + core * blend[..., None]
                            )
                            weights[ty : ty + th, tx : tx + tw] = (
                                np.asarray(weights[ty : ty + th, tx : tx + tw]) + blend
                            )
                            completed_tiles += 1
                            report("Predicting patches", completed_tiles, total_tiles)
                    # Merge connected semantic regions across tile boundaries using disk fragments.
                    import sqlite3

                    fragments = sqlite3.connect(Path(directory) / "regions.sqlite")
                    fragments.execute("CREATE TABLE pieces(class_id INTEGER,geometry BLOB)")
                    from shapely import from_wkb, to_wkb

                    report("Building regions")
                    for ty in range(0, out_height, tile_size):
                        for tx in range(0, out_width, tile_size):
                            check_cancel(cancel)
                            probabilities = np.asarray(sums[ty : ty + tile_size, tx : tx + tile_size])
                            labels = np.argmax(probabilities, axis=-1).astype(np.int32) + 1
                            for geometry, value in shapes(
                                labels, transform=Affine(ds, 0, x0 + tx * ds, 0, ds, y0 + ty * ds)
                            ):
                                clipped = shape(geometry).intersection(box(x0, y0, x1, y1))
                                fragments.execute("INSERT INTO pieces VALUES(?,?)", (int(value) - 1, to_wkb(clipped)))
                    fragments.execute("CREATE INDEX pieces_class ON pieces(class_id)")
                    fragments.commit()
                    report("Merging regions", 0, len(metadata["classes"]))
                    for class_index, class_id in enumerate(metadata["classes"]):
                        check_cancel(cancel)
                        # Hierarchical unions keep individual batches bounded. A large tissue object is resident once.
                        pending = []
                        merged = None
                        for (wkb,) in fragments.execute("SELECT geometry FROM pieces WHERE class_id=?", (class_index,)):
                            pending.append(from_wkb(wkb))
                            if len(pending) == 128:
                                merged = unary_union([*pending, *([merged] if merged is not None else [])])
                                pending.clear()
                        merged = unary_union([*pending, *([merged] if merged is not None else [])])
                        parts = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
                        for geometry in parts:
                            if geometry.is_empty:
                                continue
                            count += 1
                            store.append_staged(
                                set_id,
                                [
                                    AnnotationObject(
                                        set_id,
                                        count,
                                        geometry,
                                        class_id,
                                        provenance=provenance,
                                        review="proposed",
                                    )
                                ],
                            )
                        report("Merging regions", class_index + 1, len(metadata["classes"]))
                    fragments.close()
            else:
                report("Loading model")
                segmenter = None
                if request.backend == "micro-sam":
                    model, segmenter = _automatic_sam(request)
                else:
                    model = _pretrained(
                        request.backend, request.model, request.checkpoint, request.device, request.allow_download
                    )
                ds, tile = request.downsample, request.tile_size
                halo = min(request.halo, tile // 3)
                overview = reader.read(
                    (x0, y0, x1, y1),
                    downsample=max(ds, math.sqrt((x1 - x0) * (y1 - y0) / 262144)),
                    channels=request.channels,
                )
                ranges = np.percentile(overview.data, (1, 99), axis=(0, 1))
                provenance.update(
                    model=_model_provenance(model, request.backend),
                    preprocessing={"percentile_ranges": ranges.tolist(), "output": "RGB uint8"},
                )
                # Reconcile overlapping instances against already staged neighboring geometry using the RTree.
                total_tiles = math.ceil((y1 - int(y0)) / (tile * ds)) * math.ceil((x1 - int(x0)) / (tile * ds))
                completed_tiles = 0
                report("Segmenting patches", 0, total_tiles)
                for y in range(int(y0), math.ceil(y1), tile * ds):
                    for x in range(int(x0), math.ceil(x1), tile * ds):
                        check_cancel(cancel)
                        core = box(x, y, min(x1, x + tile * ds), min(y1, y + tile * ds))
                        patch = reader.read(
                            (
                                max(x0, x - halo * ds),
                                max(y0, y - halo * ds),
                                min(x1, x + (tile + halo) * ds),
                                min(y1, y + (tile + halo) * ds),
                            ),
                            downsample=ds,
                            channels=request.channels,
                        )
                        rgb = _rgb(patch.data, ranges)
                        masks = (
                            _sam_instance_masks(segmenter, rgb)
                            if segmenter is not None
                            else np.asarray(model.eval(rgb, channel_axis=-1, do_3D=False, batch_size=1)[0])
                        )
                        check_cancel(cancel)
                        grouped: dict[int, list] = {}
                        for geometry, label in shapes(masks.astype(np.int32), mask=masks > 0):
                            grouped.setdefault(int(label), []).append(shape(geometry))
                        for pieces in grouped.values():
                            geometry = patch.coordinates.transform(unary_union(pieces), slide.geometry).intersection(
                                box(x0, y0, x1, y1)
                            )
                            if geometry.is_empty:
                                continue
                            a, b, c, d = geometry.bounds
                            rows = store.db.execute(
                                "SELECT o.id FROM bounds b CROSS JOIN objects o ON o.id=b.id WHERE o.set_id=? "
                                "AND b.minx<=? AND b.maxx>=? AND b.miny<=? AND b.maxy>=?",
                                (set_id, c, a, d, b),
                            ).fetchall()
                            matches, conflicts = [], []
                            for row in rows:
                                existing = store.get(row[0])
                                overlap = geometry.intersection(existing.geometry).area
                                if overlap <= 0:
                                    continue
                                score = overlap / geometry.union(existing.geometry).area
                                if score >= 0.5:
                                    matches.append(existing)
                                else:
                                    conflicts.append(existing)
                            flagged = bool(conflicts) or len(matches) > 1
                            if flagged:
                                with store.transaction():
                                    for existing in [*matches, *conflicts]:
                                        if not existing.properties.get("ambiguous_seam"):
                                            ambiguous += 1
                                            store._write(
                                                replace(
                                                    existing, properties={**existing.properties, "ambiguous_seam": True}
                                                ),
                                                existing.key,
                                            )
                            if len(matches) == 1 and not conflicts:
                                existing = matches[0]
                                with store.transaction():
                                    store._write(
                                        replace(existing, geometry=existing.geometry.union(geometry)), existing.key
                                    )
                            elif core.covers(geometry.centroid):
                                count += 1
                                ambiguous += int(flagged)
                                store.append_staged(
                                    set_id,
                                    [
                                        AnnotationObject(
                                            set_id,
                                            count,
                                            geometry,
                                            request.class_id,
                                            properties={"ambiguous_seam": flagged},
                                            provenance=provenance,
                                            review="proposed",
                                        )
                                    ],
                                )
                        completed_tiles += 1
                        report("Segmenting patches", completed_tiles, total_tiles)
        report("Publishing annotations")
        check_cancel(cancel)
        if fingerprint(source_path) != slide.fingerprint:
            raise ValueError("Source image changed during prediction; results were discarded.")  # noqa: TRY301
        with store.transaction():
            store.db.execute("UPDATE annotation_sets SET metadata=? WHERE id=?", (json.dumps(provenance), set_id))
        store.publish_set(set_id, prediction=True, expected_revision=request.revision)
    except BaseException:
        store.discard_set(set_id)
        raise
    return PredictionBatch(id=request.id, set_id=set_id, request=request, count=count, ambiguous=ambiguous)
