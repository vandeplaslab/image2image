"""Real tissue training plus deterministic pretrained adapter and seam tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import tifffile
from shapely import from_wkb
from shapely.geometry import box

from image2image.models.annotation import (
    AnnotationClass,
    AnnotationObject,
    EditCommand,
    FeatureSettings,
    PredictionRequest,
)
from image2image.utils.annotation import ml
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


def test_ignore_masks_precede_conflict_checks_and_respect_sets(trained: tuple) -> None:
    store, slide, set_id, _model, settings, dark, light = trained
    ignore = AnnotationClass(name="Renamed Ignore", is_ignore=True)
    store.save_class(ignore)
    other_set = store.create_set(slide.id, "Other annotations")
    mask_geometry = box(10, 10, 30, 70)
    keys = store.apply(
        EditCommand(
            "Ignore masks",
            [
                (None, AnnotationObject(set_id, "mask", mask_geometry, ignore.id, training=False)),
                (None, AnnotationObject(other_set, "foreign mask", box(0, 0, 160, 96), ignore.id, training=True)),
                (None, AnnotationObject(set_id, "proposed mask", box(0, 0, 160, 96), ignore.id, review="proposed")),
                (None, AnnotationObject(set_id, "masked conflict", box(12, 12, 20, 25), light.id, training=True)),
            ],
        )
    )
    model = train_classifier(store, "Masked", settings)
    _forest, metadata = load_forest(model)
    assert set(metadata["classes"]) == {dark.id, light.id}
    assert [mask["key"] for mask in metadata["ignore_regions"]] == [keys[0]]
    examples = {example["id"]: example for example in metadata["training"]}
    assert "masked conflict" not in examples
    assert from_wkb(bytes.fromhex(examples["dark"]["geometry"])).equals(box(30, 10, 50, 70))
    assert from_wkb(bytes.fromhex(examples["light"]["geometry"])).equals(box(100, 10, 140, 70))


def test_training_reports_fully_masked_classes(trained: tuple) -> None:
    store, _slide, set_id, _model, settings, _dark, _light = trained
    ignore = AnnotationClass(name="Ignore", is_ignore=True)
    store.save_class(ignore)
    store.apply(
        EditCommand(
            "Mask all light pixels",
            [
                (None, AnnotationObject(set_id, "mask", box(90, 0, 160, 96), ignore.id)),
            ],
        )
    )
    with pytest.raises(ValueError, match="no usable pixels"):
        train_classifier(store, "Empty class", settings)


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
    assert calls[0]["box"] is None
    assert calls[1]["point_coords"] is None
    np.testing.assert_array_equal(calls[1]["box"], [2, 2, 14, 12])
    assert batch.count == 2
    obj = next(store.iter_objects(set_id=batch.set_id, review="proposed"))
    assert obj.geometry.bounds == (28, 36, 44, 50)
    assert obj.provenance["model"]["segmentation_mode"] == "prompts"


def test_sam_separate_prompt_instances_negative_hints_and_cancel(
    trained: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, slide, _set, _model, _settings, _dark, _light = trained

    class Predictor:
        images = 0

        def __init__(self) -> None:
            self.calls = []

        def set_image(self, image: np.ndarray) -> None:
            self.images += 1
            self.shape = image.shape[:2]

        def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, None]:
            self.calls.append(kwargs)
            masks = np.zeros((1, *self.shape), dtype=bool)
            if kwargs["box"] is not None:
                x0, y0, x1, y1 = np.asarray(kwargs["box"], dtype=int)
            else:
                x0, y0 = np.asarray(kwargs["point_coords"], dtype=int)[0]
                x1, y1 = x0 + 4, y0 + 3
            masks[0, y0:y1, x0:x1] = True
            return masks, np.array([1.0]), None

    predictor = Predictor()
    monkeypatch.setattr(ml, "_pretrained", lambda *args: predictor)
    request = PredictionRequest(
        project=str(store.path),
        slide_id=slide.id,
        revision=store.revision,
        backend="micro-sam",
        bounds=(20, 10, 100, 90),
        downsample=2,
        points=[(30, 20, 1), (60, 40, 1), (50, 30, 0)],
        boxes=[(22, 12, 34, 24), (70, 50, 90, 70)],
    )
    batch = predict(store, request)
    assert batch.count == 4 and predictor.images == 1
    objects = list(store.iter_objects(set_id=batch.set_id, review="proposed"))
    assert [obj.geometry.bounds for obj in objects] == [
        (30, 20, 38, 26),
        (60, 40, 68, 46),
        (22, 12, 34, 24),
        (70, 50, 90, 70),
    ]
    assert [list(call["point_labels"]) for call in predictor.calls] == [[1, 0], [1, 0], [0], [0]]
    assert all(obj.provenance["prompt"] for obj in objects)
    before = store.sets(slide.id)
    predictor.calls.clear()
    with pytest.raises(InterruptedError):
        predict(store, request, cancel=lambda: len(predictor.calls) > 0)
    assert store.sets(slide.id) == before
    with pytest.raises(ValueError, match="foreground point or box"):
        predict(store, request.model_copy(update={"points": [(50, 30, 0)], "boxes": []}))


@pytest.mark.parametrize("backend", ["cellpose", "micro-sam"])
@pytest.mark.parametrize("downsample,bounds", [(1, None), (2, (16, 8, 112, 88))])
def test_instance_seam_reconciliation_and_cancel(trained, monkeypatch, tmp_path, backend, downsample, bounds):
    store, _slide, _set, _model, _settings, dark, _light = trained
    data = np.zeros((96, 128), dtype=np.uint8)
    data[20:40, 56:74] = 200
    path = tmp_path / "seam.tiff"
    tifffile.imwrite(path, data)
    slide = inspect_slide(path)
    store.save_slide(slide)

    class Model:
        def eval(self, image, **_kwargs):
            return (image[..., 0] > 100).astype(np.int32), None, None

    monkeypatch.setattr("image2image.utils.annotation.ml._pretrained", lambda *args: Model())

    class Segmenter:
        def initialize(self, image: np.ndarray, **_kwargs: object) -> None:
            self.image = image

        def generate(self, **_kwargs: object) -> np.ndarray:
            return (self.image[..., 0] > 100).astype(np.int32)

    monkeypatch.setattr(ml, "_automatic_sam", lambda request: (Model(), Segmenter()))
    request = PredictionRequest(
        project=str(store.path),
        slide_id=slide.id,
        revision=store.revision,
        backend=backend,
        sam_mode="automatic",
        class_id=dark.id,
        tile_size=64,
        halo=16,
        downsample=downsample,
        bounds=bounds,
    )
    progress = []
    batch = predict(store, request, task_progress=lambda *args: progress.append(args))
    stages = [step for step in progress if step[0] == "Segmenting patches"]
    assert stages[0][1] == 0
    assert stages[-1][1] == stages[-1][2] == (4 if downsample == 1 else 1)
    assert batch.count == 1
    obj = next(store.iter_objects(set_id=batch.set_id, review="proposed"))
    assert obj.geometry.equals(box(56, 20, 74, 40))
    assert not obj.properties["ambiguous_seam"]
    before = store.sets(slide.id)
    with pytest.raises(InterruptedError):
        predict(store, request, cancel=lambda: True)
    assert store.sets(slide.id) == before
    assert store.db.execute("SELECT count(*) FROM annotation_sets WHERE state='staging'").fetchone()[0] == 0
    with pytest.raises(InterruptedError):
        store.review(batch.set_id, accept=True, cancel=lambda: True)
    assert store.count(batch.set_id) == 0


def test_seed_depth_reproducibility_and_legacy_artifacts(trained: tuple, tmp_path: Path) -> None:
    store, _slide, _set, _model, settings, _dark, _light = trained
    configured = settings.model_copy(update={"seed": 19, "max_depth": 2})
    first = train_classifier(store, "First", configured)
    second = train_classifier(store, "Second", configured)
    forest, metadata = load_forest(first)
    repeated, _metadata = load_forest(second)
    assert metadata["settings"]["seed"] == 19
    assert metadata["settings"]["max_depth"] == forest.max_depth == 2
    for left, right in zip(forest.estimators_, repeated.estimators_, strict=True):
        assert left.tree_.max_depth <= 2
        np.testing.assert_array_equal(left.tree_.threshold, right.tree_.threshold)
    with np.load(first, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    del metadata["settings"]["max_depth"]
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(legacy, **arrays)
    loaded, metadata = load_forest(legacy)
    assert loaded.max_depth == FeatureSettings.model_validate(metadata["settings"]).max_depth == 24


def test_automatic_sam_invalidates_prompt_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    class Segmenter:
        def initialize(self, image: np.ndarray, **_kwargs: object) -> None:
            self.image = image

        def generate(self, **_kwargs: object) -> np.ndarray:
            return np.ones(self.image.shape[:2], dtype=np.int32)

    monkeypatch.setattr(ml, "_SAM_IMAGE_KEY", "previous prompt embedding")
    masks = ml._sam_instance_masks(Segmenter(), np.zeros((8, 12, 3), dtype=np.uint8))
    assert masks.shape == (8, 12)
    assert ml._SAM_IMAGE_KEY == ""


@pytest.mark.parametrize("decoder_available", [False, True])
def test_automatic_sam_uses_decoder_or_default_mask_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decoder_available: bool
) -> None:
    checkpoint = tmp_path / "local.pt"
    checkpoint.write_bytes(b"local weights")
    predictor = SimpleNamespace()
    state = {"decoder_state": {}} if decoder_available else {}
    monkeypatch.setattr(ml, "_sam_bundle", lambda *args: (predictor, state))
    monkeypatch.setattr(ml, "_device", lambda device: device)
    automatic = ModuleType("micro_sam.automatic_segmentation")
    calls = []
    segmenter = object()

    def prepare(*args: object, **kwargs: object) -> tuple[object, object]:
        calls.append((args, kwargs))
        return predictor, segmenter

    automatic.get_predictor_and_segmenter = prepare
    monkeypatch.setitem(sys.modules, "micro_sam", ModuleType("micro_sam"))
    monkeypatch.setitem(sys.modules, "micro_sam.automatic_segmentation", automatic)
    request = PredictionRequest(
        project="unused",
        slide_id="slide",
        revision=0,
        backend="micro-sam",
        checkpoint=str(checkpoint),
        sam_mode="automatic",
    )
    actual_predictor, actual_segmenter = ml._automatic_sam(request)
    assert actual_predictor is predictor and actual_segmenter is segmenter
    assert calls[0][1]["segmentation_mode"] == ("ais" if decoder_available else "amg")
    assert calls[0][1]["is_tiled"] is False
    assert calls[0][1]["state"] is state


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
