"""Bounded instance datasets and explicit, cancellable pretrained-model fine-tuning."""

from __future__ import annotations

import json
import math
import random
import shutil
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import tifffile
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box

from image2image.models.annotation import FineTuneSettings
from image2image.utils.annotation.images import PatchReader, fingerprint
from image2image.utils.annotation.io import Cancel, check_cancel
from image2image.utils.annotation.model_cache import TaskProgress, resolve_pretrained_model
from image2image.utils.annotation.store import AnnotationStore


def list_finetuned_models(project: Path, backend: str) -> list[dict[str, Any]]:
    """Read project checkpoint manifests without importing model libraries or loading weights."""
    results = []
    for path in (project / "models").glob("*/metadata.json"):
        try:
            entry = json.loads(path.read_text())
            checkpoint = path.parent / "weights.pt"
            settings = FineTuneSettings.model_validate(entry["settings"])
            if (
                entry.get("kind") == "finetuned"
                and entry.get("schema_version") == 1
                and isinstance(entry.get("name"), str)
                and settings.backend == backend
                and checkpoint.is_file()
            ):
                results.append({**entry, "checkpoint": str(checkpoint), "manifest": str(path)})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(results, key=lambda entry: Path(entry["manifest"]).stat().st_mtime_ns, reverse=True)


def prepare_finetuning_patches(
    store: AnnotationStore,
    settings: FineTuneSettings,
    directory: Path,
    *,
    cancel: Cancel | None = None,
    task_progress: TaskProgress | None = None,
) -> list[dict[str, Any]]:
    """Write nonoverlapping instance crops; skip Ignore regions and reserve zero for background."""
    from image2image.utils.annotation.ml import _rgb

    classes = {item.id: item for item in store.classes()}
    if not settings.class_ids or set(settings.class_ids) - classes.keys():
        raise ValueError("Choose existing instance classes for fine-tuning.")
    selected = {key for key in settings.class_ids if not classes[key].is_ignore}
    if not selected or not 1 <= len(settings.channels) <= 3:
        raise ValueError("Choose at least one non-Ignore instance class and one to three image channels.")
    directory.mkdir(parents=True, exist_ok=True)
    patches, seen = [], set()
    size, ds = settings.patch_size, settings.downsample
    for slide in store.slides():
        check_cancel(cancel)
        candidates = store.iter_objects(slide_id=slide.id, training=True)
        with PatchReader(store.resolve_path(slide), series=slide.series) as reader:
            if fingerprint(store.resolve_path(slide)) != slide.fingerprint:
                raise ValueError("A training image changed. Reload its metadata before fine-tuning.")
            for obj in candidates:
                check_cancel(cancel)
                if obj.class_id not in selected:
                    continue
                point = obj.geometry.representative_point()
                tx, ty = int(point.x) // (size * ds), int(point.y) // (size * ds)
                tile = obj.set_id, tx, ty
                if tile in seen:
                    continue
                seen.add(tile)
                bounds = tx * size * ds, ty * size * ds, (tx + 1) * size * ds, (ty + 1) * size * ds
                region = box(*bounds)
                objects = list(store.iter_objects(set_id=obj.set_id, bounds=bounds))
                if any(
                    classes.get(item.class_id)
                    and classes[item.class_id].is_ignore
                    and item.geometry.intersection(region).area > 0
                    for item in objects
                ):
                    continue
                examples = [item for item in objects if item.training and item.class_id in selected]
                patch = reader.read(bounds, downsample=ds, channels=settings.channels)
                transform = Affine(patch.downsample[0], 0, patch.offset[0], 0, patch.downsample[1], patch.offset[1])
                labels = np.zeros(patch.data.shape[:2], dtype=np.uint32)
                for label, item in enumerate(examples, 1):
                    mask = (
                        rasterize([(item.geometry, 1)], out_shape=labels.shape, transform=transform, dtype=np.uint8)
                        != 0
                    )
                    if np.any(mask & (labels != 0)):
                        raise ValueError("Fine-tuning instances overlap. Correct overlapping objects within each set.")
                    labels[mask] = label
                if not labels.any() or not np.any(labels == 0):
                    continue
                ranges = np.percentile(patch.data, (1, 99), axis=(0, 1))
                image = _rgb(patch.data, ranges)
                if not image.any():
                    continue
                _, labels = np.unique(labels, return_inverse=True)
                labels = labels.reshape(patch.data.shape[:2]).astype(np.uint32)
                height, width = labels.shape
                if height > size or width > size:
                    raise ValueError("The source patch exceeds the configured training size.")
                image = np.pad(image, ((0, size - height), (0, size - width), (0, 0)))
                labels = np.pad(labels, ((0, size - height), (0, size - width)))
                prefix = directory / f"patch-{len(patches):05d}"
                image_path, label_path = prefix.with_suffix(".tif"), prefix.with_name(prefix.name + "-labels.tif")
                tifffile.imwrite(image_path, image, photometric="rgb")
                tifffile.imwrite(label_path, labels)
                patches.append(
                    {
                        "image": str(image_path),
                        "labels": str(label_path),
                        "slide_id": slide.id,
                        "set_id": obj.set_id,
                        "bounds": bounds,
                        "source_fingerprint": slide.fingerprint,
                        "objects": [
                            {"key": item.key, "id": item.object_id, "geometry": item.geometry.wkb_hex}
                            for item in examples
                        ],
                    }
                )
                if task_progress:
                    task_progress(
                        "Preparing fine-tuning crops", len(patches), None, f"Prepared {len(patches)} instance crops"
                    )
                if len(patches) >= settings.max_patches:
                    return patches
    return patches


class _PatchDataset:
    def __init__(self, patches: list[dict[str, Any]], cancel: Cancel | None = None) -> None:
        self.patches, self.cancel = patches, cancel

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch

        check_cancel(self.cancel)
        patch = self.patches[index]
        image = torch.from_numpy(tifffile.imread(patch["image"]).transpose(2, 0, 1).copy()).float()
        labels = torch.from_numpy(tifffile.imread(patch["labels"]).astype(np.int64)[None]).float()
        return image, labels

    def __getstate__(self) -> dict[str, Any]:
        return {"patches": self.patches}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.patches, self.cancel = state["patches"], None


def _train_cellpose(
    settings: FineTuneSettings,
    checkpoint: str,
    train: list[dict],
    validation: list[dict],
    directory: Path,
    cancel: Cancel | None,
    progress: TaskProgress | None,
) -> Path:
    from cellpose.dynamics import labels_to_flows
    from cellpose.models import CellposeModel
    from cellpose.train import train_seg

    from image2image.utils.annotation.ml import _device

    model = CellposeModel(pretrained_model=checkpoint, device=_device(settings.device))
    total, completed = settings.epochs * math.ceil(len(train) / settings.batch_size), 0

    def before_forward(network: Any, _inputs: Any) -> None:
        nonlocal completed
        check_cancel(cancel)
        if network.training:
            completed += 1
            if progress:
                progress("Fine-tuning Cellpose", completed - 1, total, f"Training batch {completed}/{total}")

    hook = model.net.register_forward_pre_hook(before_forward)
    try:

        def arrays(patches: list[dict], key: str) -> list[np.ndarray]:
            values = []
            for patch in patches:
                check_cancel(cancel)
                array = tifffile.imread(patch[key])
                values.append(labels_to_flows([array], device=model.net.device)[0] if key == "labels" else array)
            return values

        path, train_loss, val_loss = train_seg(
            model.net,
            train_data=arrays(train, "image"),
            train_labels=arrays(train, "labels"),
            test_data=arrays(validation, "image"),
            test_labels=arrays(validation, "labels"),
            channel_axis=-1,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            n_epochs=settings.epochs,
            min_train_masks=1,
            save_path=str(directory),
            model_name="weights.pt",
        )
        (directory / "losses.json").write_text(
            json.dumps({"training": np.asarray(train_loss).tolist(), "validation": np.asarray(val_loss).tolist()})
        )
        if progress:
            progress("Fine-tuning Cellpose", total, total, "Training and validation complete")
        return Path(path)
    finally:
        hook.remove()


def _train_sam(
    settings: FineTuneSettings,
    checkpoint: str,
    train: list[dict],
    validation: list[dict],
    directory: Path,
    cancel: Cancel | None,
    progress: TaskProgress | None,
) -> Path:
    from micro_sam.training import train_sam
    from micro_sam.util import export_custom_sam_model
    from torch.utils.data import DataLoader

    total, completed = settings.epochs * math.ceil(len(train) / settings.batch_size), 0

    def update(steps: int) -> None:
        nonlocal completed
        check_cancel(cancel)
        completed += steps
        if progress:
            progress("Fine-tuning SAM", completed, total, f"Training iteration {completed}/{total}")

    signals = SimpleNamespace(
        pbar_total=SimpleNamespace(emit=lambda _value: check_cancel(cancel)),
        pbar_update=SimpleNamespace(emit=update),
        pbar_description=SimpleNamespace(emit=lambda _value: check_cancel(cancel)),
    )
    train_loader = DataLoader(_PatchDataset(train, cancel), batch_size=settings.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(_PatchDataset(validation, cancel), batch_size=1, num_workers=0)
    train_sam(
        name="finetune",
        model_type=settings.model[:5],
        checkpoint_path=checkpoint,
        train_loader=train_loader,
        val_loader=validation_loader,
        n_epochs=settings.epochs,
        early_stopping=None,
        n_objects_per_batch=4,
        with_segmentation_decoder=False,
        freeze=["image_encoder"] if settings.freeze_encoder else None,
        device=settings.device,
        lr=settings.learning_rate,
        n_sub_iteration=2,
        save_root=str(directory),
        pbar_signals=signals,
    )
    check_cancel(cancel)
    output = directory / "weights.pt"
    export_custom_sam_model(
        checkpoint_path=str(directory / "checkpoints" / "finetune" / "best.pt"),
        model_type=settings.model[:5],
        save_path=str(output),
    )
    return output


def fine_tune_model(
    store: AnnotationStore,
    name: str,
    settings: FineTuneSettings,
    *,
    cancel: Cancel | None = None,
    task_progress: TaskProgress | None = None,
) -> dict[str, Any]:
    """Fine-tune a fresh backend model and atomically publish weights and their input manifest."""
    import torch

    from image2image.utils.annotation.ml import _device

    check_cancel(cancel)
    checkpoint = settings.checkpoint or resolve_pretrained_model(settings.backend, settings.model)[0]
    if not Path(checkpoint).is_file():
        raise ValueError("Choose a local checkpoint or download the base model before fine-tuning.")
    device = _device(settings.device)
    if settings.backend == "micro-sam" and device.type == "mps":
        if settings.device != "auto":
            raise ValueError("SAM training supports CPU or CUDA. Select CPU or Auto for fine-tuning on this machine.")
        device = _device("cpu")
    settings = settings.model_copy(update={"device": str(device)})
    revision, base_fingerprint = store.revision, fingerprint(Path(checkpoint))
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    with TemporaryDirectory(prefix="finetune-", dir=store.path / "staging") as temporary:
        directory = Path(temporary)
        patches = prepare_finetuning_patches(
            store, settings, directory / "data", cancel=cancel, task_progress=task_progress
        )
        if len(patches) < 2:
            raise ValueError(
                "Fine-tuning needs at least two usable annotated crops for training and validation. Include instances in separate patches; avoid Ignore masks and provide background around objects."
            )
        groups: dict[tuple[str, tuple[float, ...]], list[int]] = {}
        for index, patch in enumerate(patches):
            groups.setdefault((patch["slide_id"], tuple(patch["bounds"])), []).append(index)
        if len(groups) < 2:
            raise ValueError("Fine-tuning needs at least two spatially distinct crops for training and validation.")
        grouped = list(groups.values())
        order = np.random.default_rng(settings.seed).permutation(len(grouped))
        validation_indices = {index for group in order[: max(1, len(grouped) // 5)] for index in grouped[group]}
        validation = [patch for index, patch in enumerate(patches) if index in validation_indices]
        train = [patch for index, patch in enumerate(patches) if index not in validation_indices]
        check_cancel(cancel)
        if task_progress:
            task_progress(
                "Loading fine-tuning model",
                0,
                None,
                f"Loading {settings.model}; {len(train)} training and {len(validation)} validation crops",
            )
        trainer = _train_cellpose if settings.backend == "cellpose" else _train_sam
        from image2image.utils.annotation import ml

        ml._load_pretrained.cache_clear()
        ml._sam_bundle.cache_clear()
        ml._SAM_IMAGE_KEY = ""
        weights = trainer(settings, checkpoint, train, validation, directory, cancel, task_progress)
        check_cancel(cancel)
        if store.revision != revision or fingerprint(Path(checkpoint)) != base_fingerprint:
            raise ValueError("Fine-tuning inputs changed. The checkpoint was discarded; run training again.")
        for slide_id in {patch["slide_id"] for patch in patches}:
            slide = store.slide(slide_id)
            if fingerprint(store.resolve_path(slide)) != slide.fingerprint:
                raise ValueError("A source image changed during fine-tuning. The checkpoint was discarded.")
        if task_progress:
            task_progress("Validating fine-tuned checkpoint", 0, None, "Checking trained weights before publishing")
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(state, dict) or not state:
            raise ValueError("The backend did not produce valid model weights.")
        parameters = state.get("model_state", state)
        if not isinstance(parameters, dict) or not any(
            isinstance(value, torch.Tensor) for value in parameters.values()
        ):
            raise ValueError("The backend checkpoint contains no model tensors.")
        for value in parameters.values():
            check_cancel(cancel)
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("Training produced non-finite weights. Lower the learning rate and train again.")
        del state, parameters, value
        publish = directory / "publish"
        publish.mkdir()
        shutil.move(str(weights), publish / "weights.pt")
        if (directory / "losses.json").exists():
            shutil.move(str(directory / "losses.json"), publish / "losses.json")
        metadata = {
            "kind": "finetuned",
            "schema_version": 1,
            "name": name.strip() or "Fine-tuned model",
            "settings": settings.model_dump(),
            "revision": revision,
            "base_fingerprint": base_fingerprint,
            "patches": [
                {key: value for key, value in patch.items() if key not in ("image", "labels")} for patch in patches
            ],
            "validation_indices": sorted(validation_indices),
            "sam_automatic_mode": "amg" if settings.backend == "micro-sam" else None,
            "versions": {
                name: version(name)
                for name in ("torch", "numpy", "micro-sam" if settings.backend == "micro-sam" else "cellpose")
            },
            "backend_parameters": {"n_objects_per_batch": 4, "n_sub_iteration": 2, "with_segmentation_decoder": False}
            if settings.backend == "micro-sam"
            else {"min_train_masks": 1, "optimizer": "AdamW", "weight_decay": 0.1},
        }
        (publish / "metadata.json").write_text(json.dumps(metadata, indent=2))
        check_cancel(cancel)
        slug = "".join(char if char.isalnum() or char in "-_" else "_" for char in metadata["name"])
        destination = store.path / "models" / f"{slug[:60] or 'finetuned'}-{uuid4().hex[:8]}"
        with store.transaction():
            check_cancel(cancel)
            if store.revision != revision or fingerprint(Path(checkpoint)) != base_fingerprint:
                raise ValueError("Fine-tuning inputs changed before publishing. The checkpoint was discarded.")
            publish.replace(destination)
    return {**metadata, "checkpoint": str(destination / "weights.pt"), "manifest": str(destination / "metadata.json")}
