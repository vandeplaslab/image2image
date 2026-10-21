"""Validated polygon edits with explicit survivor IDs and lineage."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import split, unary_union

from image2image.models.annotation import AnnotationObject, EditCommand
from image2image.utils.annotation.store import AnnotationStore, validate_geometry


def edit_geometry(store: AnnotationStore, key: int, geometry: BaseGeometry, label: str = "Edit geometry") -> None:
    """Replace complete object geometry without changing its ID or metadata."""
    validate_geometry(geometry)
    obj = store.get(key)
    store.apply(EditCommand(label, [(key, replace(obj, geometry=geometry))]))


def brush_object(store: AnnotationStore, key: int, stroke: BaseGeometry, *, subtract: bool = False) -> None:
    """Add/subtract a stroke, retaining holes and components; delete fully erased objects."""
    obj = store.get(key)
    geometry = obj.geometry.difference(stroke) if subtract else obj.geometry.union(stroke)
    if geometry.is_empty:
        store.apply(EditCommand("Erase object", [(key, None)]))
        return
    edit_geometry(store, key, geometry, "Erase from object" if subtract else "Brush object")


def brush_stroke(points: list[tuple[float, float]], radius: float, *, fill: bool = False) -> Polygon:
    """Buffer a continuous brush stroke, optionally filling its enclosed interiors."""
    stroke = (LineString(points) if len(points) > 1 else Point(points[0])).buffer(radius)
    return Polygon(stroke.exterior) if fill else stroke


def split_preview(obj: AnnotationObject, cut: LineString) -> list[BaseGeometry]:
    """Return valid split pieces, largest first, without changing the project."""
    pieces = list(split(obj.geometry, cut).geoms)
    pieces = [piece for piece in pieces if not piece.is_empty and piece.area > 0]
    if len(pieces) <= (len(obj.geometry.geoms) if obj.geometry.geom_type == "MultiPolygon" else 1):
        raise ValueError("The cut must cross the interior of the object.")
    for piece in pieces:
        validate_geometry(piece)
    return sorted(pieces, key=lambda piece: (-piece.area, *piece.bounds))


def split_object(store: AnnotationStore, key: int, cut: LineString, survivor: int = 0) -> list[int]:
    """Split an object; the selected piece keeps its ID and new pieces record their parent."""
    obj = store.get(key)
    pieces = split_preview(obj, cut)
    if not 0 <= survivor < len(pieces):
        raise ValueError("Invalid survivor piece.")
    next_id = int(store.next_id(obj.set_id, integer=True)) if isinstance(obj.object_id, int) else None
    changes = []
    for index, geometry in enumerate(pieces):
        retained = index == survivor
        object_id = obj.object_id if retained else next_id if next_id is not None else uuid4().hex
        if not retained and next_id is not None:
            next_id += 1
        lineage = [*obj.provenance.get("lineage", []), {"operation": "split", "parents": [obj.object_id]}]
        changes.append(
            (
                key if retained else None,
                replace(
                    obj,
                    key=key if retained else 0,
                    geometry=geometry,
                    object_id=object_id,
                    provenance={**obj.provenance, "lineage": lineage},
                ),
            )
        )
    return store.apply(EditCommand("Split object", changes))


def merge_objects(store: AnnotationStore, keys: list[int], survivor: int, class_id: str | None = None) -> int:
    """Union selected objects within one set while retaining an explicitly chosen ID."""
    if len(set(keys)) < 2 or survivor not in keys:
        raise ValueError("Select at least two objects and a survivor.")
    objects = [store.get(key) for key in dict.fromkeys(keys)]
    if len({obj.set_id for obj in objects}) != 1:
        raise ValueError("Merge is limited to objects in one annotation set.")
    retained = next(obj for obj in objects if obj.key == survivor)
    if class_id is None and len({obj.class_id for obj in objects}) > 1:
        raise ValueError("Choose a class when merging objects with different classes.")
    geometry = unary_union([obj.geometry for obj in objects])
    validate_geometry(geometry)
    parents = [obj.object_id for obj in objects]
    lineage = [*retained.provenance.get("lineage", []), {"operation": "merge", "parents": parents}]
    provenance = {**retained.provenance, "lineage": lineage}
    review_inputs = []
    for obj in objects:
        for inputs in [obj.provenance.get("review_inputs"), *obj.provenance.get("merged_review_inputs", [])]:
            if inputs and inputs not in review_inputs:
                review_inputs.append(inputs)
    if review_inputs:
        provenance["merged_review_inputs"] = review_inputs
    merged = replace(
        retained,
        geometry=geometry,
        class_id=class_id or retained.class_id,
        provenance=provenance,
    )
    store.apply(EditCommand("Merge objects", [(obj.key, merged if obj.key == survivor else None) for obj in objects]))
    return survivor
