"""Opt-in real pretrained inference; downloads weights only when explicitly enabled."""

from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

from image2image.models.annotation import PredictionRequest
from image2image.utils.annotation.images import inspect_slide
from image2image.utils.annotation.ml import predict
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
