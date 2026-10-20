"""Opt-in real pretrained inference; downloads weights only when explicitly enabled."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
from shapely.geometry import Point

from image2image.models.annotation import (
    AnnotationClass,
    AnnotationObject,
    EditCommand,
    FineTuneSettings,
    PredictionRequest,
)
from image2image.utils.annotation.finetune import fine_tune_model
from image2image.utils.annotation.images import inspect_slide
from image2image.utils.annotation.ml import predict
from image2image.utils.annotation.model_cache import resolve_pretrained_model
from image2image.utils.annotation.store import AnnotationStore

pytestmark = pytest.mark.skipif(
    os.environ.get("I2I_ANNOTATION_REAL_MODELS") != "1", reason="Enable real model downloads explicitly"
)


@pytest.mark.parametrize(
    "backend,sam_mode", [("micro-sam", "prompts"), ("micro-sam", "automatic"), ("cellpose", "prompts")]
)
def test_real_pretrained_inference(tmp_path, backend, sam_mode):
    import torch
    from skimage.data import cells3d, immunohistochemistry

    torch.set_num_threads(4)
    image = immunohistochemistry()[100:228, 100:228]
    if backend == "cellpose":
        image = np.repeat(cells3d()[30, 1, 64:192, 64:192, None], 3, axis=-1)
    source = tmp_path / "histology.tiff"
    tifffile.imwrite(source, image, photometric="rgb")
    with AnnotationStore(tmp_path / "smoke.i2annotate", create=True) as store:
        slide = inspect_slide(source)
        store.save_slide(slide)
        request = PredictionRequest(
            project=str(store.path),
            slide_id=slide.id,
            revision=store.revision,
            backend=backend,
            sam_mode=sam_mode,
            device="cpu",
            channels=(0, 1, 2),
            tile_size=128,
            halo=16,
            allow_download=True,
            model="vit_b_lm" if backend == "micro-sam" else "cpsam",
            points=[(64, 64, 1)] if backend == "micro-sam" else [],
            box=(30, 30, 100, 100) if backend == "micro-sam" else None,
        )
        batch = predict(store, request)
        assert batch.count > 0
        assert store.count(batch.set_id) == 0
        for obj in store.iter_objects(set_id=batch.set_id, review="proposed"):
            assert obj.geometry.is_valid and obj.geometry.area > 0
            assert obj.provenance["request"]["backend"] == backend
        store.review(batch.set_id, accept=True)
        assert store.count(batch.set_id) == batch.count


@pytest.mark.skipif(
    os.environ.get("I2I_ANNOTATION_REAL_TRAINING") != "1", reason="Enable full-model training explicitly"
)
@pytest.mark.parametrize("backend", ["cellpose", "micro-sam"])
def test_real_finetuning_round_trip(tmp_path: Path, backend: str) -> None:
    import torch

    torch.set_num_threads(4)
    rng = np.random.default_rng(42)
    image = rng.normal(20, 3, (512, 512)).astype(np.float32)
    geometries = [
        Point(x + tx, y + ty).buffer(12)
        for tx in (0, 256)
        for ty in (0, 256)
        for x, y in ((60, 60), (160, 60), (100, 160))
    ]
    yy, xx = np.indices(image.shape)
    for geometry in geometries:
        center = geometry.centroid
        image += 150 * np.exp(-((xx - center.x) ** 2 + (yy - center.y) ** 2) / 100)
    source = tmp_path / "instances.tif"
    tifffile.imwrite(source, np.clip(image, 0, 255).astype(np.uint8))
    model = "vit_b_lm" if backend == "micro-sam" else "cpsam"
    checkpoint, _decoder = resolve_pretrained_model(backend, model, allow_download=True)
    with AnnotationStore(tmp_path / "training.i2annotate", create=True) as store:
        slide = inspect_slide(source)
        store.save_slide(slide)
        cells = AnnotationClass(name="Cells")
        store.save_class(cells)
        set_id = store.create_set(slide.id, "Instances")
        store.apply(
            EditCommand(
                "Training masks",
                [
                    (None, AnnotationObject(set_id, index, geometry, cells.id, training=True))
                    for index, geometry in enumerate(geometries)
                ],
            )
        )
        settings = FineTuneSettings(
            backend=backend,
            model=model,
            checkpoint=checkpoint,
            class_ids=(cells.id,),
            device="cpu",
            epochs=2,
            max_patches=4,
        )
        result = fine_tune_model(store, "Smoke model", settings)
        assert Path(result["checkpoint"]).is_file() and len(result["patches"]) == 4
        request = PredictionRequest(
            project=str(store.path),
            slide_id=slide.id,
            revision=store.revision,
            backend=backend,
            model=model,
            checkpoint=result["checkpoint"],
            device="cpu",
            sam_mode="prompts" if backend == "micro-sam" else "automatic",
            points=[(60, 60, 1)] if backend == "micro-sam" else [],
            bounds=(0, 0, 256, 256),
            class_id=cells.id,
        )
        batch = predict(store, request)
        assert batch.set_id
        assert all(obj.geometry.is_valid for obj in store.iter_objects(set_id=batch.set_id, review="proposed"))
