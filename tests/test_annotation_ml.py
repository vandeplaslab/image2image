"""Real tissue training plus deterministic pretrained adapter and seam tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from shapely.geometry import box

from image2image.models.annotation import (
    AnnotationClass,
    AnnotationObject,
    EditCommand,
    FeatureSettings,
    PredictionRequest,
)
from image2image.utils.annotation.images import inspect_slide
from image2image.utils.annotation.ml import load_forest, predict, train_classifier
from image2image.utils.annotation.store import AnnotationStore


@pytest.fixture
def trained(tmp_path: Path):
    image = np.zeros((96, 160), dtype=np.uint8)
    image[:, 80:] = 220
    path = tmp_path / "training.tiff"
    tifffile.imwrite(path, image)
    with AnnotationStore(tmp_path / "project.i2annotate", create=True) as store:
        slide = inspect_slide(path)
        store.save_slide(slide)
        dark, light = AnnotationClass(name="Dark"), AnnotationClass(name="Light")
        store.save_class(dark)
        store.save_class(light)
        set_id = store.create_set(slide.id, "Training")
        store.apply(
            EditCommand(
                "Examples",
                [
                    (None, AnnotationObject(set_id, "dark", box(10, 10, 50, 70), dark.id, training=True)),
                    (None, AnnotationObject(set_id, "light", box(100, 10, 140, 70), light.id, training=True)),
                ],
            )
        )
        settings = FeatureSettings(downsample=1, sigma_max=2, samples_per_class=100, trees=10)
        model = train_classifier(store, "Test", settings)
        yield store, slide, set_id, model, settings, dark, light


def test_forest_save_load_predict_review_and_seams(trained):
    store, slide, _set, model, _settings, dark, _light = trained
    _forest, metadata = load_forest(model)
    assert metadata["samples_per_class"] == 100
    request = PredictionRequest(
        project=str(store.path),
        slide_id=slide.id,
        revision=store.revision,
        model=str(model),
        downsample=1,
        tile_size=64,
        halo=8,
    )
    batch = predict(store, request)
    assert batch.count == 2
    objects = list(store.iter_objects(set_id=batch.set_id, review="proposed"))
    assert all(obj.geometry.is_valid for obj in objects)
    assert sum(obj.geometry.area for obj in objects) == 96 * 160
    for obj in objects:
        expected = box(0, 0, 80, 96) if obj.class_id == dark.id else box(80, 0, 160, 96)
        assert obj.geometry.symmetric_difference(expected).area < 96 * 3
    assert store.count(batch.set_id) == 0
    store.review(batch.set_id, accept=True)
    assert store.count(batch.set_id) == 2
    store.undo()
    assert store.count(batch.set_id) == 0


def test_training_conflicts_and_stale_requests(trained):
    store, slide, set_id, model, settings, _dark, light = trained
    request = PredictionRequest(project=str(store.path), slide_id=slide.id, revision=store.revision, model=str(model))
    store.apply(
        EditCommand("Conflict", [(None, AnnotationObject(set_id, "bad", box(11, 11, 25, 25), light.id, training=True))])
    )
    with pytest.raises(ValueError, match="overlap"):
        train_classifier(store, "Bad", settings)
    with pytest.raises(ValueError, match="stale"):
        predict(store, request)


def test_missing_backend_and_cancellation(trained, monkeypatch):
    store, slide, _set, _model, _settings, _dark, _light = trained
    monkeypatch.setattr("image2image.utils.annotation.ml.backend_available", lambda _name: False)
    request = PredictionRequest(
        project=str(store.path), slide_id=slide.id, revision=store.revision, backend="cellpose", allow_download=True
    )
    with pytest.raises(ImportError, match="annotate-cellpose"):
        predict(store, request)
    assert all(item["state"] == "ready" for item in store.sets(slide.id))
    with pytest.raises(InterruptedError):
        train_classifier(store, "Cancel", FeatureSettings(), cancel=lambda: True)


def test_sam_prompts_patch_coordinates_and_cache(trained, monkeypatch):
    store, slide, _set, _model, _settings, _dark, _light = trained
    calls = []

    class Predictor:
        def set_image(self, rgb):
            self.shape = rgb.shape[:2]

        def predict(self, **kwargs):
            calls.append(kwargs)
            masks = np.zeros((1, *self.shape), dtype=bool)
            masks[0, 3:10, 4:12] = True
            return masks, np.array([0.9]), None

    predictor = Predictor()
    monkeypatch.setattr("image2image.utils.annotation.ml._pretrained", lambda *args: predictor)
    request = PredictionRequest(
        project=str(store.path),
        slide_id=slide.id,
        revision=store.revision,
        backend="micro-sam",
        bounds=(20, 30, 60, 70),
        points=[(30, 40, 1)],
        box=(24, 34, 48, 54),
        downsample=2,
    )
    batch = predict(store, request)
    np.testing.assert_array_equal(calls[0]["point_coords"], [[5, 5]])
    np.testing.assert_array_equal(calls[0]["box"], [2, 2, 14, 12])
    obj = next(store.iter_objects(set_id=batch.set_id, review="proposed"))
    assert obj.geometry.bounds == (28, 36, 44, 50)


def test_instance_seam_reconciliation_and_cancel(trained, monkeypatch, tmp_path):
    store, _slide, _set, _model, _settings, dark, _light = trained
    data = np.zeros((96, 128), dtype=np.uint8)
    data[20:40, 55:74] = 200
    path = tmp_path / "seam.tiff"
    tifffile.imwrite(path, data)
    slide = inspect_slide(path)
    store.save_slide(slide)

    class Model:
        def eval(self, image, **_kwargs):
            return (image[..., 0] > 100).astype(np.int32), None, None

    monkeypatch.setattr("image2image.utils.annotation.ml._pretrained", lambda *args: Model())
    request = PredictionRequest(
        project=str(store.path),
        slide_id=slide.id,
        revision=store.revision,
        backend="cellpose",
        class_id=dark.id,
        tile_size=64,
        halo=16,
    )
    batch = predict(store, request)
    assert batch.count == 1
    obj = next(store.iter_objects(set_id=batch.set_id, review="proposed"))
    assert obj.geometry.equals(box(55, 20, 74, 40))
    assert not obj.properties["ambiguous_seam"]
    before = store.sets(slide.id)
    with pytest.raises(InterruptedError):
        predict(store, request, cancel=lambda: True)
    assert store.sets(slide.id) == before
    assert store.db.execute("SELECT count(*) FROM annotation_sets WHERE state='staging'").fetchone()[0] == 0
    with pytest.raises(InterruptedError):
        store.review(batch.set_id, accept=True, cancel=lambda: True)
    assert store.count(batch.set_id) == 0


def test_ambiguous_instance_seams_flag_both_objects(trained, monkeypatch):
    store, slide, _set, _model, _settings, _dark, _light = trained

    class Model:
        calls = 0

        def eval(self, image, **_kwargs):
            masks = np.zeros(image.shape[:2], dtype=np.int32)
            if self.calls == 0:
                masks[10:40, 50:75] = 1
            elif self.calls == 1:
                masks[10:40, 22:42] = 1
            self.calls += 1
            return masks, None, None

    model = Model()
    monkeypatch.setattr("image2image.utils.annotation.ml._pretrained", lambda *args: model)
    request = PredictionRequest(
        project=str(store.path), slide_id=slide.id, revision=store.revision, backend="cellpose", tile_size=64, halo=16
    )
    batch = predict(store, request)
    assert batch.count == batch.ambiguous == 2
    assert all(obj.properties["ambiguous_seam"] for obj in store.iter_objects(set_id=batch.set_id, review="proposed"))
