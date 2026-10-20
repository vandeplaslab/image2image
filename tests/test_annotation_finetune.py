"""Instance crop preparation, training adapters, and atomic checkpoint publication."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tifffile
from shapely.geometry import box

from image2image.models.annotation import AnnotationClass, AnnotationObject, EditCommand, FineTuneSettings
from image2image.utils.annotation import finetune
from image2image.utils.annotation.images import inspect_slide
from image2image.utils.annotation.store import AnnotationStore
from image2image.utils.annotation.worker import run_job


@pytest.fixture
def instance_project(tmp_path: Path) -> tuple:
    image = tmp_path / "cells.tif"
    yy, xx = np.indices((512, 600))
    tifffile.imwrite(image, ((xx + yy) % 256).astype(np.uint8))
    with AnnotationStore(tmp_path / "instances.i2annotate", create=True) as store:
        slide = inspect_slide(image)
        store.save_slide(slide)
        cells, other, ignore = (
            AnnotationClass(name="Cells"),
            AnnotationClass(name="Other"),
            AnnotationClass(name="Ignore", is_ignore=True),
        )
        for item in (cells, other, ignore):
            store.save_class(item)
        set_id = store.create_set(slide.id, "Instances")
        objects = [
            AnnotationObject(set_id, index, box(x, y, x + 30, y + 25), cells.id, training=True)
            for index, (x, y) in enumerate(((30, 30), (280, 30), (540, 30), (30, 300), (280, 300), (540, 300)))
        ]
        store.apply(EditCommand("Instances", [(None, obj) for obj in objects]))
        checkpoint = tmp_path / "base.pt"
        checkpoint.write_bytes(b"base")
        settings = FineTuneSettings(
            backend="cellpose",
            model="cpsam",
            checkpoint=str(checkpoint),
            class_ids=(cells.id,),
            downsample=1,
            epochs=2,
            max_patches=6,
            device="cpu",
        )
        yield store, slide, set_id, cells, other, ignore, settings


def test_instance_crops_masks_padding_and_exclusions(instance_project: tuple, tmp_path: Path) -> None:
    store, slide, set_id, cells, other, ignore, settings = instance_project
    store.apply(
        EditCommand(
            "Extra regions",
            [
                (None, AnnotationObject(set_id, "ignored", box(0, 0, 80, 80), ignore.id)),
                (None, AnnotationObject(set_id, "excluded", box(285, 35, 290, 40), other.id, training=True)),
                (
                    None,
                    AnnotationObject(
                        set_id, "proposal", box(550, 35, 555, 40), cells.id, review="proposed", training=True
                    ),
                ),
            ],
        )
    )
    patches = finetune.prepare_finetuning_patches(store, settings, tmp_path / "data")
    assert len(patches) == 5
    for patch in patches:
        image, labels = tifffile.imread(patch["image"]), tifffile.imread(patch["labels"])
        assert image.shape == (256, 256, 3) and labels.shape == (256, 256)
        assert list(np.unique(labels)) == [0, 1]
        assert patch["slide_id"] == slide.id
    edge = next(patch for patch in patches if patch["bounds"][0] == 512)
    assert not tifffile.imread(edge["labels"])[:, 88:].any()
    assert all(len(patch["objects"]) == 1 for patch in patches)
    with pytest.raises(InterruptedError):
        finetune.prepare_finetuning_patches(store, settings, tmp_path / "cancelled", cancel=lambda: True)


@pytest.mark.parametrize("backend", ["cellpose", "micro-sam"])
def test_finetune_publication_and_manifest_reuse(
    instance_project: tuple, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    torch = pytest.importorskip("torch")
    store, _slide, _set, _cells, _other, _ignore, settings = instance_project
    settings = settings.model_copy(
        update={"backend": backend, "model": "vit_b_lm" if backend == "micro-sam" else "cpsam"}
    )
    seen = []

    def train(config, checkpoint, train, validation, directory, cancel, progress):
        assert len(train) == 5 and len(validation) == 1
        assert not {item["image"] for item in train}.intersection(item["image"] for item in validation)
        assert config.epochs == 2 and checkpoint == settings.checkpoint
        path = directory / "trained.pt"
        torch.save({"weight": torch.tensor([1.0])}, path)
        seen.extend([*train, *validation])
        return path

    monkeypatch.setattr(finetune, "_train_cellpose" if backend == "cellpose" else "_train_sam", train)
    monkeypatch.setattr(finetune, "version", lambda _name: "test")
    result = finetune.fine_tune_model(store, "Custom cells", settings)
    assert Path(result["checkpoint"]).is_file()
    assert all(not Path(item["image"]).exists() for item in seen)
    assert not list((store.path / "staging").iterdir())
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["settings"]["class_ids"] == list(settings.class_ids)
    assert manifest["settings"]["epochs"] == 2 and manifest["validation_indices"]
    assert len(manifest["patches"]) == 6 and all("image" not in item for item in manifest["patches"])
    assert finetune.list_finetuned_models(store.path, backend)[0]["checkpoint"] == result["checkpoint"]
    assert finetune.list_finetuned_models(store.path, "cellpose" if backend == "micro-sam" else "micro-sam") == []
    Path(result["checkpoint"]).unlink()
    assert not finetune.list_finetuned_models(store.path, backend)


@pytest.mark.parametrize("failure", ["cancel", "revision", "source", "checkpoint"])
def test_failed_finetuning_never_publishes(
    instance_project: tuple, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    torch = pytest.importorskip("torch")
    store, slide, set_id, cells, _other, _ignore, settings = instance_project
    cancelled = False

    def train(config, checkpoint, train, validation, directory, cancel, progress):
        nonlocal cancelled
        path = directory / "weights.pt"
        torch.save({"weight": torch.tensor([1.0])}, path)
        if failure == "cancel":
            cancelled = True
        elif failure == "revision":
            store.apply(EditCommand("Changed", [(None, AnnotationObject(set_id, "new", box(2, 2, 4, 4), cells.id))]))
        elif failure == "source":
            with Path(slide.path).open("ab") as stream:
                stream.write(b"changed")
        else:
            Path(checkpoint).write_bytes(b"changed")
        return path

    monkeypatch.setattr(finetune, "_train_cellpose", train)
    with pytest.raises((InterruptedError, ValueError), match=r"cancel|changed"):
        finetune.fine_tune_model(store, "Failed", settings, cancel=lambda: cancelled)
    assert not list((store.path / "models").iterdir())
    assert not list((store.path / "staging").iterdir())


def test_sam_adapter_inputs_export_and_cancel(
    instance_project: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    training = pytest.importorskip("micro_sam.training")
    util = pytest.importorskip("micro_sam.util")
    store, _slide, _set, _cells, _other, _ignore, settings = instance_project
    settings = settings.model_copy(update={"backend": "micro-sam", "model": "vit_b_lm"})
    patches = finetune.prepare_finetuning_patches(store, settings, tmp_path / "patches")
    seen = []

    def train(**kwargs):
        assert kwargs["model_type"] == "vit_b" and kwargs["with_segmentation_decoder"] is False
        assert kwargs["freeze"] == ["image_encoder"] and kwargs["lr"] == 1e-5
        for loader in (kwargs["train_loader"], kwargs["val_loader"]):
            image, labels = next(iter(loader))
            assert image.shape[1:] == (3, 256, 256) and labels.shape[1:] == (1, 256, 256)
            assert 0 <= image.min() < image.max() <= 255
            pickle.loads(pickle.dumps(loader.dataset))  # noqa: S301 - round-trip this test's own dataset
        kwargs["pbar_signals"].pbar_update.emit(1)
        folder = Path(kwargs["save_root"]) / "checkpoints" / "finetune"
        folder.mkdir(parents=True)
        torch.save({"model_state": {"sam.weight": torch.tensor([2.0])}}, folder / "best.pt")

    def export(**kwargs):
        state = torch.load(kwargs["checkpoint_path"], weights_only=True)
        torch.save(state["model_state"], kwargs["save_path"])

    monkeypatch.setattr(training, "train_sam", train)
    monkeypatch.setattr(util, "export_custom_sam_model", export)
    output = finetune._train_sam(
        settings, settings.checkpoint, patches[:4], patches[4:], tmp_path, None, lambda *args: seen.append(args)
    )
    assert output.is_file() and seen[0][0] == "Fine-tuning SAM"
    with pytest.raises(InterruptedError):
        finetune._PatchDataset(patches, lambda: True)[0]


def test_worker_finetune_dispatch_and_custom_discovery(
    instance_project: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _slide, _set, _cells, _other, _ignore, settings = instance_project
    seen = []
    monkeypatch.setattr(
        finetune,
        "fine_tune_model",
        lambda store, name, config, **kwargs: seen.append((name, config)) or {"checkpoint": "custom"},
    )
    event = SimpleNamespace(is_set=lambda: False)
    result = run_job(
        {"operation": "finetune", "project": str(store.path), "name": "Cells", "settings": settings.model_dump()},
        event,
        None,
    )
    assert result["checkpoint"] == "custom" and seen == [("Cells", settings)]
    artifact = store.path / "models" / "custom"
    artifact.mkdir()
    (artifact / "weights.pt").write_bytes(b"complete")
    metadata = {
        "kind": "finetuned",
        "schema_version": 1,
        "name": "Custom DINO",
        "settings": settings.model_copy(update={"model": "cpdino"}).model_dump(),
    }
    (artifact / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(
        "image2image.utils.annotation.model_cache.list_pretrained_models",
        lambda _backend: [{"name": "cpdino", "reason": "Requires DINOv3", "cached": False}],
    )
    catalogue = run_job(
        {"operation": "model_catalogue", "backend": "cellpose", "project": str(store.path)}, event, None
    )
    assert catalogue[-1]["custom"]["name"] == "Custom DINO"
    assert catalogue[-1]["reason"] == "Requires DINOv3"


def test_duplicate_crops_stay_in_one_validation_partition(
    instance_project: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    store, slide, source, _cells, _other, _ignore, settings = instance_project
    from dataclasses import replace

    duplicate = store.create_set(slide.id, "Duplicate annotations")
    store.apply(
        EditCommand(
            "Copy", [(None, replace(obj, key=0, set_id=duplicate)) for obj in store.iter_objects(set_id=source)]
        )
    )

    def train(config, checkpoint, train, validation, directory, cancel, progress):
        training_locations = {(patch["slide_id"], tuple(patch["bounds"])) for patch in train}
        validation_locations = {(patch["slide_id"], tuple(patch["bounds"])) for patch in validation}
        assert not training_locations.intersection(validation_locations)
        assert len(train) + len(validation) == 12
        output = directory / "weights.pt"
        torch.save({"weight": torch.tensor([1.0])}, output)
        return output

    monkeypatch.setattr(finetune, "_train_cellpose", train)
    monkeypatch.setattr(finetune, "version", lambda _name: "test")
    result = finetune.fine_tune_model(store, "Grouped", settings.model_copy(update={"max_patches": 12}))
    assert len(result["validation_indices"]) == 2


def test_nonfinite_weights_are_not_published(instance_project: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    store, _slide, _set, _cells, _other, _ignore, settings = instance_project

    def train(config, checkpoint, train, validation, directory, cancel, progress):
        output = directory / "weights.pt"
        torch.save({"weight": torch.tensor([float("nan")])}, output)
        return output

    monkeypatch.setattr(finetune, "_train_cellpose", train)
    with pytest.raises(ValueError, match="non-finite"):
        finetune.fine_tune_model(store, "Invalid", settings)
    assert not list((store.path / "models").iterdir())


def test_cellpose_adapter_runs_actual_optimizer_with_small_network(
    instance_project: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("cellpose.models")
    store, _slide, _set, _cells, _other, _ignore, settings = instance_project
    patches = finetune.prepare_finetuning_patches(store, settings, tmp_path / "patches")

    class SmallNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device, self.dtype, self.backbone = torch.device("cpu"), torch.float32, "sam_vitl"
            self.conv = torch.nn.Conv2d(3, 3, 1)
            self.diam_mean = torch.nn.Parameter(torch.tensor(30.0), requires_grad=False)
            self.diam_labels = torch.nn.Parameter(torch.tensor(30.0), requires_grad=False)

        def forward(self, image: Any) -> tuple[Any, None]:
            return self.conv(image), None

        def save_model(self, path: str | Path) -> None:
            torch.save(self.state_dict(), path)

    network = SmallNetwork()
    before = network.conv.weight.detach().clone()
    monkeypatch.setattr(models, "CellposeModel", lambda **kwargs: SimpleNamespace(net=network))
    progress = []
    output = finetune._train_cellpose(
        settings, settings.checkpoint, patches[:1], patches[1:2], tmp_path, None, lambda *args: progress.append(args)
    )
    assert output.is_file() and not network._forward_pre_hooks
    assert not torch.equal(before, network.conv.weight)
    assert progress[-1][2] == 2
    losses = json.loads((tmp_path / "losses.json").read_text())
    assert len(losses["training"]) == 2 and np.isfinite(losses["training"]).all()
