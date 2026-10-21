"""Lossless identity, coordinate conversion, transactions, and tiled annotation IO."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from image2image.models.annotation import AnnotationClass, AnnotationObject, CoordinateSpec, EditCommand, SlideGeometry
from image2image.utils.annotation.geometry import brush_object, brush_stroke, edit_geometry, merge_objects, split_object
from image2image.utils.annotation.images import PatchReader, inspect_slide
from image2image.utils.annotation.io import export_geojson, export_labels, import_geojson, import_labels
from image2image.utils.annotation.render import AnnotationRenderer
from image2image.utils.annotation.store import AnnotationStore


@pytest.fixture
def project(tmp_path: Path):
    image = tmp_path / "image.ome.tiff"
    tifffile.imwrite(
        image, np.zeros((96, 128), dtype=np.uint8), metadata={"axes": "YX", "PhysicalSizeX": 0.4, "PhysicalSizeY": 0.7}
    )
    with AnnotationStore(tmp_path / "test.i2annotate", create=True) as store:
        slide = inspect_slide(image)
        store.save_slide(slide)
        class_ = AnnotationClass(name="Tissue", color="#00ff00")
        store.save_class(class_)
        set_id = store.create_set(slide.id, "Manual")
        yield store, slide, class_, set_id


def add(store, set_id, geometry=None, object_id=23, class_id=None, **kwargs):
    obj = AnnotationObject(
        set_id, object_id, geometry if geometry is not None else box(0, 0, 20, 20), class_id, **kwargs
    )
    return store.apply(EditCommand("Add", [(None, obj)]))[0]


def test_split_merge_identity_history_reopen(project):
    store, slide, cls, set_id = project
    key = add(store, set_id, object_id=2**60, class_id=cls.id, properties={"custom": "keep"})
    keys = split_object(store, key, LineString([(7, -1), (7, 25)]), survivor=1)
    assert len(keys) == 2
    assert store.get(key).object_id == 2**60
    assert store.get(key).geometry.area == 140
    assert {obj.object_id for obj in store.iter_objects(set_id=set_id)} == {2**60, 2**60 + 1}
    merge_objects(store, keys, key)
    assert store.count(set_id) == 1
    assert store.get(key).geometry.area == 400
    assert store.get(key).properties == {"custom": "keep"}
    assert store.get(key).provenance["lineage"][-1]["parents"] == [2**60 + 1, 2**60]
    with AnnotationStore(store.path) as reopened:
        reopened.undo()
        assert reopened.count(set_id) == 2
        reopened.undo()
        assert reopened.get(key).geometry.equals(box(0, 0, 20, 20))
        reopened.redo()
        assert reopened.count(set_id) == 2
    assert store.hit_test(slide.id, (2, 2)).key == key


def test_area_selection_filters_review_set_and_visibility(project: tuple) -> None:
    store, slide, cls, set_id = project
    accepted = add(store, set_id, object_id="accepted", class_id=cls.id)
    proposed = add(store, set_id, object_id="proposed", class_id=cls.id, review="proposed")
    add(store, set_id, object_id="rejected", review="rejected")
    hidden = AnnotationClass(name="Hidden", visible=False)
    store.save_class(hidden)
    add(store, set_id, object_id="hidden", class_id=hidden.id)
    other_set = store.create_set(slide.id, "Other")
    add(store, other_set, object_id="other")
    add(store, set_id, geometry=box(40, 40, 50, 50), object_id="outside")
    region = box(10, 10, 30, 30)
    assert list(store.selection_in(slide.id, region, set_id=set_id)) == [accepted]
    assert list(store.selection_in(slide.id, region, set_id=set_id, review="proposed")) == [proposed]


def test_set_counts_combined_copy_and_remove_undo(project: tuple) -> None:
    store, slide, cls, first = project
    second = store.create_set(slide.id, "Second")
    left = add(store, first, object_id=12, class_id=cls.id, training=True)
    right = add(store, second, object_id=12, class_id=cls.id, review="proposed")
    add(store, second, object_id="12", class_id=cls.id, review="rejected")
    assert [info["object_count"] for info in store.sets(slide.id)] == [1, 1]
    combined = store.combine_sets([first, second], "Combined")
    objects = list(store.iter_objects(set_id=combined, review=None))
    assert len(objects) == 3
    assert len({(type(obj.object_id), obj.object_id) for obj in objects}) == 3
    assert store.get(left).object_id == store.get(right).object_id == 12
    assert {obj.review for obj in objects} == {"accepted", "proposed", "rejected"}
    assert next(obj for obj in objects if obj.review == "accepted").training
    assert {obj.provenance["combined_from"]["set_id"] for obj in objects} == {first, second}
    assert next(info for info in store.sets(slide.id) if info["id"] == combined)["object_count"] == 2
    store.remove_set(first)
    assert first not in {info["id"] for info in store.sets(slide.id)}
    assert not list(store.iter_objects(set_id=first))
    assert store.get(left).object_id == 12
    assert store.undo() == "Remove annotation set"
    assert first in {info["id"] for info in store.sets(slide.id)}
    assert store.count(first) == 1
    store.redo()
    assert first not in {info["id"] for info in store.sets(slide.id)}


def test_combined_copy_cancellation_and_same_slide_validation(project: tuple) -> None:
    store, slide, cls, first = project
    store.apply(
        EditCommand("Many", [(None, AnnotationObject(first, index, box(1, 1, 3, 3), cls.id)) for index in range(501)])
    )
    second = store.create_set(slide.id, "Second")
    before = store.sets(slide.id)
    cancelled = False

    def progress(*_args: object) -> None:
        nonlocal cancelled
        cancelled = True

    with pytest.raises(InterruptedError):
        store.combine_sets([first, second], "Cancelled", cancel=lambda: cancelled, task_progress=progress)
    assert store.sets(slide.id) == before
    assert store.db.execute("SELECT count(*) FROM annotation_sets WHERE state='staging'").fetchone()[0] == 0
    other_slide = slide.model_copy(update={"id": "other"})
    store.save_slide(other_slide)
    other_set = store.create_set(other_slide.id, "Other slide")
    with pytest.raises(ValueError, match="same slide"):
        store.combine_sets([first, other_set], "Wrong slide")


def test_combined_proposals_keep_review_revision_and_model_validation(project: tuple, tmp_path: Path) -> None:
    from image2image.utils.annotation.images import fingerprint

    store, slide, cls, _manual = project
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"original")
    metadata = {
        "request": {"revision": store.revision},
        "source_fingerprint": slide.fingerprint,
        "model": {"checkpoint": str(checkpoint), "fingerprint": fingerprint(checkpoint)},
    }
    sources = []
    for index in range(2):
        source = store.create_set(slide.id, f"Prediction {index}", staging=True, metadata=metadata)
        store.append_staged(
            source, [AnnotationObject(source, index, box(index * 10, 0, index * 10 + 5, 5), cls.id, review="proposed")]
        )
        store.publish_set(source, prediction=True)
        sources.append(source)
    combined = store.combine_sets(sources, "Review combined")
    keys = [obj.key for obj in store.iter_objects(set_id=combined, review="proposed")]
    store.review(combined, accept=True, keys=keys[:1])
    assert store.count(combined) == 1
    store.review(combined, accept=True, keys=keys[1:])
    assert store.count(combined) == 2
    assert all(store.count(source, review="proposed") == 1 for source in sources)
    # Copying stale sources never makes them valid for acceptance again.
    stale = store.combine_sets(sources, "Stale originals")
    with pytest.raises(ValueError, match="stale before combining"):
        store.review(stale, accept=True)
    store.undo()  # Undo the stale copy; the accepted combined batch remains available.
    # A fresh combined batch still checks the original model asset at acceptance.
    current = store.revision
    with store.transaction():
        for source in sources:
            values = store.set_info(source)["metadata"]
            values["review_revision"] = current
            store.db.execute("UPDATE annotation_sets SET metadata=? WHERE id=?", (json.dumps(values), source))
    fresh = store.combine_sets(sources, "Fresh copy")
    checkpoint.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="checkpoint changed"):
        store.review(fresh, accept=True)


def test_merged_combined_proposals_validate_every_model(project: tuple, tmp_path: Path) -> None:
    from image2image.utils.annotation.images import fingerprint

    store, slide, cls, _manual = project
    sources = []
    checkpoints = []
    for index in range(2):
        checkpoint = tmp_path / f"checkpoint-{index}.bin"
        checkpoint.write_bytes(b"original")
        checkpoints.append(checkpoint)
        source = store.create_set(
            slide.id,
            f"Prediction {index}",
            staging=True,
            metadata={
                "request": {"revision": store.revision},
                "source_fingerprint": slide.fingerprint,
                "model": {"checkpoint": str(checkpoint), "fingerprint": fingerprint(checkpoint)},
            },
        )
        store.append_staged(
            source, [AnnotationObject(source, index, box(index, 0, index + 2, 2), cls.id, review="proposed")]
        )
        store.publish_set(source, prediction=True)
        sources.append(source)
    combined = store.combine_sets(sources, "Combined")
    keys = [obj.key for obj in store.iter_objects(set_id=combined, review="proposed")]
    merge_objects(store, keys, keys[0])
    assert len(store.get(keys[0]).provenance["merged_review_inputs"]) == 2
    checkpoints[1].write_bytes(b"changed non-surviving model")
    with pytest.raises(ValueError, match="checkpoint changed"):
        store.review(combined, accept=True)
    assert store.count(combined, review="proposed") == 1
    store.undo()
    assert store.count(combined, review="proposed") == 2


def test_atomic_invalid_geometry_locks_and_redo_branch(project):
    store, _slide, cls, set_id = project
    key = add(store, set_id, class_id=cls.id)
    revision = store.revision
    invalid = Polygon([(0, 0), (5, 5), (0, 5), (5, 0)])
    with pytest.raises(ValueError):
        store.apply(EditCommand("Bad batch", [(key, None), (None, AnnotationObject(set_id, "bad", invalid))]))
    assert store.count(set_id) == 1 and store.revision == revision
    cls.locked = True
    store.save_class(cls)
    with pytest.raises(ValueError, match="locked"):
        edit_geometry(store, key, box(0, 0, 5, 5))
    store.undo()
    edit_geometry(store, key, box(0, 0, 5, 5))
    assert store.redo() is None
    obj = store.get(key)
    store.apply(EditCommand("Lock", [(key, replace(obj, properties={"isLocked": True}))]), allow_unlock=True)
    with pytest.raises(ValueError, match="locked"):
        edit_geometry(store, key, box(0, 0, 8, 8))
    store.apply(
        EditCommand("Unlock", [(key, replace(store.get(key), properties={"isLocked": False}))]), allow_unlock=True
    )
    edit_geometry(store, key, box(0, 0, 8, 8))
    assert store.get(key).geometry.area == 64


def test_brush_holes_and_multipart(project):
    store, _slide, _cls, set_id = project
    key = add(store, set_id)
    brush_object(store, key, box(5, 5, 10, 10), subtract=True)
    assert len(store.get(key).geometry.interiors) == 1
    brush_object(store, key, box(40, 40, 45, 45))
    assert store.get(key).geometry.geom_type == "MultiPolygon"
    assert store.get(key).object_id == 23


def test_brush_stroke_fills_only_enclosed_painted_regions(project: tuple) -> None:
    store, _slide, _cls, set_id = project
    loop = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    painted = brush_stroke(loop, 1, fill=True)
    outline = brush_stroke(loop, 1)
    assert painted.covers(box(2, 2, 8, 8)) and not painted.interiors
    assert len(outline.interiors) == 1
    assert not brush_stroke(loop[:3], 1, fill=True).covers(box(2, 2, 8, 8))
    assert brush_stroke([(5, 5)], 1, fill=True).area > 0
    original = box(-5, -5, 20, 20).difference(box(12, 12, 18, 18))
    key = add(store, set_id, geometry=original)
    brush_object(store, key, painted)
    assert len(store.get(key).geometry.interiors) == 1
    brush_object(store, key, painted, subtract=True)
    assert len(store.get(key).geometry.interiors) == 2
    store.undo()
    assert store.get(key).geometry.equals(original)


def test_explicit_coordinates():
    slide = SlideGeometry(shape=(100, 200), spacing=(0.4, 0.7), origin=(-22, 13))
    points = np.array([[0.5, 0.5], [31, 92]])
    np.testing.assert_allclose(slide.to_pixels(slide.to_physical(points)), points)
    geometry = Polygon([(1.1, 2.2), (8.7, 2.2), (8.7, 19.9)])
    for spec in (CoordinateSpec(units="um"), CoordinateSpec(offset=(120, 89), downsample=(2, 3))):
        result = spec.transform(spec.transform(geometry, slide), slide, inverse=True)
        assert result.equals_exact(geometry, 1e-12)
    with pytest.raises(ValueError, match="calibrat"):
        CoordinateSpec(units="um").transform(geometry, SlideGeometry(shape=(100, 100)))
    with pytest.raises(ValueError, match="pixel"):
        CoordinateSpec(units="um", offset=(1, 2)).transform(geometry, slide)


@pytest.mark.parametrize("units", ["px", "um"])
def test_geojson_preserves_holes_ids_properties(project, tmp_path, units):
    store, slide, cls, set_id = project
    polygon = Polygon(box(2, 2, 20, 20).exterior, [box(7, 7, 12, 12).exterior])
    geometry = MultiPolygon([polygon, box(50, 50, 55, 55)])
    add(
        store,
        set_id,
        geometry,
        object_id="000012",
        class_id=cls.id,
        properties={"name": "sample", "measurements": {"x": 4}},
    )
    path = tmp_path / "objects.geojson"
    spec = CoordinateSpec(units=units)
    export_geojson(store, set_id, path, spec)
    imported = import_geojson(store, slide.id, path, spec)
    obj = next(store.iter_objects(set_id=imported))
    assert obj.object_id == "000012"
    assert obj.geometry.equals_exact(geometry, 1e-12)
    assert obj.properties["measurements"] == {"x": 4}
    assert obj.class_id == cls.id
    assert store.find_id(imported, 12) is None


@pytest.mark.parametrize("format_", ["geojson", "labels"])
def test_exchange_training_defaults_and_ignore_role(project: tuple, tmp_path: Path, format_: str) -> None:
    store, slide, cls, set_id = project
    ignore = AnnotationClass(name="Renamed exclusion", is_ignore=True)
    store.save_class(ignore)
    add(store, set_id, box(1, 1, 8, 8), object_id=1, class_id=cls.id, training=False)
    add(store, set_id, box(10, 10, 20, 20), object_id=2, class_id=ignore.id, training=False)
    path = tmp_path / ("exchange.geojson" if format_ == "geojson" else "exchange.tiff")
    if format_ == "geojson":
        export_geojson(store, set_id, path, CoordinateSpec())
        data = json.loads(path.read_text())
        del data["features"][0]["properties"]["image2image"]["training"]
        path.write_text(json.dumps(data))
    else:
        export_labels(store, set_id, path, CoordinateSpec())
        sidecar = path.with_name(path.name + ".objects.json")
        data = json.loads(sidecar.read_text())
        del data["objects"][0]["training"]
        sidecar.write_text(json.dumps(data))
    with AnnotationStore(tmp_path / "import.i2annotate", create=True) as target:
        target.save_slide(slide)
        importer = import_geojson if format_ == "geojson" else import_labels
        imported = importer(target, slide.id, path, CoordinateSpec())
        objects = {obj.object_id: obj for obj in target.iter_objects(set_id=imported)}
        assert objects[1].training
        assert not objects[2].training
        classes = {item.id: item for item in target.classes()}
        assert classes[objects[2].class_id].is_ignore
        summary = {row["class_name"]: row for row in target.training_summary()}
        assert summary["Tissue"]["included"] == 1
        assert summary["Renamed exclusion"]["included"] == 0
        assert summary["Renamed exclusion"]["is_ignore"] == 1


def test_legacy_class_and_training_summary(project: tuple) -> None:
    store, _slide, cls, set_id = project
    assert not AnnotationClass.model_validate({"name": "Ignore"}).is_ignore
    add(store, set_id, object_id=1, class_id=cls.id, training=True)
    add(store, set_id, object_id=2, class_id=cls.id, training=False)
    add(store, set_id, object_id=3, class_id=cls.id, training=True, review="proposed")
    add(store, set_id, object_id=4, training=True)
    summary = {row["class_name"]: row for row in store.training_summary()}
    assert summary["Tissue"]["total"] == 2
    assert summary["Tissue"]["included"] == 1
    assert summary["Unclassified"]["included"] == 0


@pytest.mark.parametrize("container", ["single", "array", "collection"])
def test_geojson_forms_and_atomic_failure(project, tmp_path, container):
    store, slide, _cls, _set_id = project
    feature = {
        "type": "Feature",
        "id": 2**63 + 1,
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 0]]]},
        "properties": {},
    }
    data = (
        feature
        if container == "single"
        else [feature]
        if container == "array"
        else {"features": [feature], "type": "FeatureCollection"}
    )
    path = tmp_path / "input.geojson"
    path.write_text(json.dumps(data))
    set_id = import_geojson(store, slide.id, path, CoordinateSpec())
    assert next(store.iter_objects(set_id=set_id)).object_id == 2**63 + 1
    existing = store.sets(slide.id)
    path.write_text(json.dumps([feature, feature]))
    with pytest.raises(sqlite3.IntegrityError):
        import_geojson(store, slide.id, path, CoordinateSpec())
    assert store.sets(slide.id) == existing


@pytest.mark.parametrize("extension", [".tiff", ".ome.zarr"])
def test_integer_label_roundtrip_large_ids_seams_and_holes(project, tmp_path, extension):
    store, slide, _cls, _set_id = project
    labels = np.zeros((96, 128), dtype=np.uint64)
    labels[3:89, 4:55] = 2**63 + 7
    labels[9:21, 12:35] = 0
    labels[0, 0] = 2**63 + 7
    labels[17:81, 73:110] = 11
    source = tmp_path / "source.tiff"
    tifffile.imwrite(source, labels)
    set_id = import_labels(store, slide.id, source, CoordinateSpec(), tile_size=16)
    assert store.count(set_id) == 2
    path = tmp_path / ("output" + extension)
    export_labels(store, set_id, path, CoordinateSpec(), tile_size=16)
    if extension == ".tiff":
        np.testing.assert_array_equal(tifffile.imread(path), labels)
    else:
        import zarr

        np.testing.assert_array_equal(zarr.open(str(path), mode="r")["0"][:], labels)
    reimported = import_labels(store, slide.id, path, CoordinateSpec(), tile_size=16)
    for obj in store.iter_objects(set_id=set_id):
        assert obj.geometry.equals(store.find_id(reimported, obj.object_id).geometry)


def test_mapped_raster_ids_and_overlap_rejection(project, tmp_path):
    store, slide, cls, set_id = project
    add(store, set_id, box(3, 3, 20, 20), "cell-01", cls.id)
    with pytest.raises(ValueError, match="representable"):
        export_labels(store, set_id, tmp_path / "invalid.ome.tiff", CoordinateSpec())
    assert not (tmp_path / "invalid.ome.tiff").exists()
    path = tmp_path / "mapped.ome.tiff"
    export_labels(store, set_id, path, CoordinateSpec(), mapped_ids=True, tile_size=16)
    imported = import_labels(store, slide.id, path, CoordinateSpec(), tile_size=16)
    assert store.find_id(imported, "cell-01").class_id == cls.id
    add(store, set_id, box(10, 10, 25, 25), "cell-02", cls.id)
    with pytest.raises(ValueError, match="overlap"):
        export_labels(store, set_id, tmp_path / "overlap.tiff", CoordinateSpec(), mapped_ids=True)
    assert not (tmp_path / "overlap.tiff.objects.json").exists()


def test_cancel_and_staging_recovery(project, tmp_path):
    store, slide, _cls, set_id = project
    add(store, set_id)
    path = tmp_path / "cancel.geojson"
    with pytest.raises(InterruptedError):
        export_geojson(store, set_id, path, CoordinateSpec(), cancel=lambda: True)
    assert not path.exists()
    staged = store.create_set(slide.id, "Interrupted", staging=True)
    store.append_staged(staged, [AnnotationObject(staged, "x", box(0, 0, 1, 1))])
    assert staged not in {item["id"] for item in store.sets(slide.id)}
    assert store.recover() == 1


def test_proposal_partial_review_refine_undo_and_staleness(project):
    store, slide, _cls, set_id = project
    revision = store.revision
    proposed = store.create_set(slide.id, "Predictions", staging=True, metadata={"request": {"revision": revision}})
    store.append_staged(
        proposed, [AnnotationObject(proposed, i, box(i * 10, 0, i * 10 + 5, 5), review="proposed") for i in range(3)]
    )
    store.publish_set(proposed, prediction=True, expected_revision=revision)
    keys = [obj.key for obj in store.iter_objects(set_id=proposed, review="proposed")]
    edit_geometry(store, keys[0], box(0, 0, 4, 4))
    assert store.revision == revision
    store.review(proposed, accept=True, keys=keys[:1])
    store.undo()
    store.review(proposed, accept=True, keys=keys[:1])
    store.review(proposed, accept=True, keys=keys[1:2])
    assert store.count(proposed) == 2
    add(store, set_id, object_id="new")
    with pytest.raises(ValueError, match="stale"):
        store.review(proposed, accept=True, keys=keys[2:])
    store.review(proposed, accept=False, keys=keys[2:])


def test_image_calibration_patch_mapping_and_no_full_array(tmp_path):
    image = np.arange(120 * 180, dtype=np.uint16).reshape(120, 180)
    path = tmp_path / "sample.ome.tiff"
    tifffile.imwrite(path, image, metadata={"axes": "YX", "PhysicalSizeX": 0.3, "PhysicalSizeY": 0.8})
    assert inspect_slide(path).geometry.spacing == (0.3, 0.8)
    with PatchReader(path) as reader:
        patch = reader.read((14, 18, 44, 38), channels=(0,))
        np.testing.assert_array_equal(patch.data[..., 0], image[18:38, 14:44])
        assert patch.offset == (14, 18) and patch.downsample == (1, 1)
        small = reader.read((0, 0, 180, 120), downsample=10)
        assert small.data.shape == (12, 18, 1)
        with pytest.raises(ValueError, match="memory"):
            reader.read((0, 0, 180, 120), max_pixels=30)
    path = tmp_path / "unknown.tiff"
    tifffile.imwrite(path, image)
    assert inspect_slide(path).geometry.spacing is None


def test_renderer_cache_and_visibility(project):
    store, slide, cls, set_id = project
    add(store, set_id, class_id=cls.id)
    renderer = AnnotationRenderer(store.path, max_bytes=256 * 256 * 4)
    frame = renderer.render(slide.id, (0, 0, 100, 90), set_id=set_id)
    assert frame.rgba[5, 5, 3] > 0
    assert renderer.cache_bytes <= renderer.max_bytes
    cls.visible = False
    store.save_class(cls)
    frame = renderer.render(slide.id, (0, 0, 100, 90), set_id=set_id)
    assert not frame.rgba.any()


def test_numeric_ids_are_not_reused_after_deletion_or_undo(project):
    store, _slide, _cls, set_id = project
    key = add(store, set_id, object_id=2**64 - 3)
    store.apply(EditCommand("Delete", [(key, None)]))
    fresh = store.next_id(set_id, integer=True)
    assert fresh == 2**64 - 2
    key = add(store, set_id, object_id=fresh)
    store.undo()
    assert store.next_id(set_id, integer=True) == 2**64 - 1


def test_failed_import_does_not_publish_classes(project, tmp_path):
    store, slide, _cls, _set = project
    path = tmp_path / "bad.geojson"
    feature = {
        "type": "Feature",
        "id": "one",
        "properties": {"classification": {"name": "Temporary", "color": [1, 2, 3]}},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [5, 0], [5, 5], [0, 0]]]},
    }
    path.write_text(json.dumps([feature, feature]))
    with pytest.raises(sqlite3.IntegrityError):
        import_geojson(store, slide.id, path, CoordinateSpec())
    assert [item.name for item in store.classes()] == ["Tissue"]
    assert store.db.execute("SELECT count(*) FROM classes").fetchone()[0] == 1


def test_export_snapshot_during_concurrent_edit(project, tmp_path):
    store, _slide, _cls, set_id = project
    add(store, set_id, object_id="original")

    def edited(_count, _message):
        with AnnotationStore(store.path) as other:
            add(other, set_id, object_id="new")

    path = export_geojson(store, set_id, tmp_path / "snapshot.geojson", CoordinateSpec(), progress=edited)
    assert [feature["id"] for feature in json.loads(path.read_text())["features"]] == ["original"]
    assert store.count(set_id) == 2


def test_multichannel_patches_and_lazy_display_pyramid(tmp_path):
    import dask.array as da
    import zarr

    path = tmp_path / "channels.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    data = root.create_array("0", shape=(4, 4096, 5000), chunks=(1, 128, 128), dtype="u1", fill_value=0)
    for c in range(4):
        data[c, :128, :128] = c + 1
    root.attrs["multiscales"] = [{"axes": ["c", "y", "x"], "datasets": [{"path": "0"}]}]
    with PatchReader(path) as reader:
        patch = reader.read((16, 20, 25, 28), channels=(3, 1))
        assert patch.data.shape == (8, 9, 2)
        np.testing.assert_array_equal(patch.data[0, 0], [4, 2])
        assert all(isinstance(level, da.Array) for level in reader.display_levels)
        assert max(reader.display_levels[-1].shape[1:]) <= 1024
        assert reader.levels[0].shape == (4, 4096, 5000)
