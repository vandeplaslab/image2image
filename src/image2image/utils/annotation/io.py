"""Streaming, explicit-coordinate annotation exchange with lossless object identity."""

from __future__ import annotations

import json
import math
import os
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Self
from uuid import uuid4

import numpy as np
from shapely import from_wkb, to_wkb
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from image2image.models.annotation import AnnotationClass, AnnotationObject, CoordinateSpec
from image2image.utils.annotation.store import AnnotationStore

Progress = Callable[[int, str], None]
Cancel = Callable[[], bool]


def check_cancel(cancel: Cancel | None) -> None:
    """Stop a staged operation at a bounded work boundary."""
    if cancel is not None and cancel():
        raise InterruptedError("Operation cancelled.")


def _progress(progress: Progress | None, count: int, text: str) -> None:
    if progress:
        progress(count, text)


def iter_geojson(path: Path) -> Iterator[dict]:
    """Stream FeatureCollection, feature-array, and single-Feature GeoJSON documents."""
    import ijson

    prefix = ""
    with path.open("rb") as stream:
        for key, event, value in ijson.parse(stream):
            if key == "" and event == "start_array":
                prefix = "item"
                break
            if key == "" and event == "map_key" and value == "features":
                prefix = "features.item"
                break
            if key == "type" and event == "string":
                if value == "FeatureCollection":
                    prefix = "features.item"
                elif value != "Feature":
                    raise ValueError("GeoJSON must contain Features or a FeatureCollection.")
                break
    with path.open("rb") as stream:
        for feature in ijson.items(stream, prefix, use_float=False):
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise ValueError("GeoJSON entries must be Features.")
            # Decimal coordinate values become floats, while integer IDs stay exact.
            yield json.loads(json.dumps(feature, default=float))


def _class_for(
    store: AnnotationStore, properties: dict, cache: dict[str, str], set_id: str, *, is_ignore: bool = False
) -> str | None:
    classification = properties.get("classification")
    if not classification:
        return None
    name = classification.get("name", "Unclassified") if isinstance(classification, dict) else str(classification)
    if name not in cache:
        color = (
            classification.get("color", classification.get("colorRGB")) if isinstance(classification, dict) else None
        )
        if isinstance(color, list) and len(color) >= 3:
            color = "#" + "".join(f"{int(v):02x}" for v in color[:3])
        elif isinstance(color, int):
            color = f"#{color & 0xFFFFFF:06x}"
        if not isinstance(color, str) or not color.startswith("#"):
            color = "#e57373"
        annotation_class = AnnotationClass(name=name, color=color, is_ignore=is_ignore)
        store.stage_class(set_id, annotation_class)
        cache[name] = annotation_class.id
    return cache[name]


def import_geojson(
    store: AnnotationStore,
    slide_id: str,
    path: str | Path,
    coordinates: CoordinateSpec,
    *,
    cancel: Cancel | None = None,
    progress: Progress | None = None,
) -> str:
    """Import complete geometries and exact typed IDs into a new, atomically published set."""
    path = Path(path)
    slide = store.slide(slide_id)
    classes = {item.name: item.id for item in store.classes()}
    set_id = store.create_set(
        slide_id, path.stem, staging=True, metadata={"source": str(path), "coordinates": coordinates.model_dump()}
    )
    batch = []
    try:
        count = 0
        for feature in iter_geojson(path):
            check_cancel(cancel)
            properties = feature.get("properties") or {}
            private = properties.pop("image2image", {})
            geometry = coordinates.transform(shape(feature["geometry"]), slide.geometry)
            object_id = feature.get("id", properties.get("object_id", properties.get("id", uuid4().hex)))
            batch.append(
                AnnotationObject(
                    set_id,
                    object_id,
                    geometry,
                    _class_for(
                        store, properties, classes, set_id, is_ignore=bool(private.get("class_is_ignore", False))
                    ),
                    properties,
                    private.get("provenance", {"source": str(path)}),
                    training=bool(private.get("training", True)),
                )
            )
            if len(batch) >= 256:
                count += store.append_staged(set_id, batch)
                batch.clear()
                _progress(progress, count, f"Imported {count:,} objects")
        if batch:
            count += store.append_staged(set_id, batch)
        check_cancel(cancel)
        store.publish_set(set_id)
        _progress(progress, count, f"Imported {count:,} objects")
    except BaseException:
        store.discard_set(set_id)
        raise
    return set_id


def export_geojson(
    store: AnnotationStore,
    set_id: str,
    path: str | Path,
    coordinates: CoordinateSpec,
    *,
    cancel: Cancel | None = None,
    progress: Progress | None = None,
) -> Path:
    """Atomically stream accepted objects with classes, calibration, and original IDs."""
    with store.read_snapshot():
        path = Path(path)
        slide = store.slide(store.set_info(set_id)["slide_id"])
        classes = {item.id: item for item in store.classes()}
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w") as stream:
                metadata = {"frame": coordinates.model_dump(), "slide_geometry": slide.geometry.model_dump()}
                stream.write('{"type":"FeatureCollection","image2image":')
                json.dump(metadata, stream)
                stream.write(',"features":[')
                for count, obj in enumerate(store.iter_objects(set_id=set_id)):
                    check_cancel(cancel)
                    properties = dict(obj.properties)
                    if obj.class_id in classes:
                        item = classes[obj.class_id]
                        classification = properties.get("classification", {})
                        classification = dict(classification) if isinstance(classification, dict) else {}
                        classification.update(name=item.name, color=[int(item.color[i : i + 2], 16) for i in (1, 3, 5)])
                        properties["classification"] = classification
                    properties.setdefault("objectType", "annotation")
                    properties["image2image"] = {
                        "provenance": obj.provenance,
                        "training": obj.training,
                        "class_is_ignore": bool(obj.class_id in classes and classes[obj.class_id].is_ignore),
                    }
                    feature = {
                        "type": "Feature",
                        "id": obj.object_id,
                        "geometry": mapping(coordinates.transform(obj.geometry, slide.geometry, inverse=True)),
                        "properties": properties,
                    }
                    if count:
                        stream.write(",")
                    json.dump(feature, stream, ensure_ascii=False, allow_nan=False)
                    if count % 256 == 0:
                        _progress(progress, count, "Exporting GeoJSON")
                stream.write("]}")
            check_cancel(cancel)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


class LabelSource:
    """Lazy two-dimensional integer TIFF or Zarr labels without interpolation."""

    def __init__(self, path: str | Path) -> None:
        import dask.array as da
        import zarr

        self.path = Path(path)
        self.tif = None
        self.store = None
        self.metadata: dict = {}
        if self.path.is_dir():
            root = zarr.open(str(self.path), mode="r")
            if isinstance(root, zarr.Array):
                self.array = da.from_zarr(root)  # type: ignore[no-untyped-call]
            else:
                attrs: dict[str, Any] = root.attrs.asdict()
                attrs = attrs.get("ome", attrs)
                level = attrs.get("multiscales", [{"datasets": [{"path": "0"}]}])[0]["datasets"][0]["path"]
                self.array = da.from_zarr(root[level])  # type: ignore[no-untyped-call]
                self.metadata = attrs.get("image2image", {})
        else:
            import tifffile

            self.tif = tifffile.TiffFile(self.path)
            self.store = self.tif.series[0].aszarr()
            self.array = da.from_zarr(zarr.open(self.store, mode="r"))  # type: ignore[no-untyped-call]
        self.array = self.array.squeeze()
        if self.array.ndim != 2 or not np.issubdtype(self.array.dtype, np.integer):
            self.close()
            raise ValueError("Label images must be 2D integer arrays, with zero as background.")

    def close(self) -> None:
        """Release label file and lazy Zarr adapter handles."""
        if self.store is not None:
            self.store.close()
        if self.tif is not None:
            self.tif.close()

    def __enter__(self) -> Self:
        """Keep the lazy label file open during a streaming operation."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the label reader at context exit."""
        self.close()


def _label_metadata(path: Path) -> Iterator[dict]:
    import ijson

    sidecar = path.with_name(path.name + ".objects.json")
    if sidecar.exists():
        with sidecar.open("rb") as stream:
            for item in ijson.items(stream, "objects.item", use_float=False):
                yield json.loads(json.dumps(item, default=float))


def import_labels(
    store: AnnotationStore,
    slide_id: str,
    path: str | Path,
    coordinates: CoordinateSpec,
    *,
    tile_size: int = 512,
    cancel: Cancel | None = None,
    progress: Progress | None = None,
) -> str:
    """Polygonize bounded tiles, reunite fragments by label ID, and publish atomically."""
    from rasterio.features import shapes
    from rasterio.transform import Affine

    if coordinates.units != "px":
        raise ValueError("Raster labels use a pixel grid; specify pixel offset/downsample and image calibration.")
    path = Path(path)
    slide = store.slide(slide_id)
    set_id = store.create_set(
        slide_id, path.stem, staging=True, metadata={"source": str(path), "coordinates": coordinates.model_dump()}
    )
    classes = {item.name: item.id for item in store.classes()}
    try:
        with TemporaryDirectory(dir=store.path / "staging") as directory, LabelSource(path) as labels:
            fragments = sqlite3.connect(Path(directory) / "fragments.sqlite")
            try:
                fragments.executescript(
                    "CREATE TABLE pieces(id TEXT,geometry BLOB);CREATE INDEX pieces_id ON pieces(id);"
                    "CREATE TABLE metadata(id TEXT PRIMARY KEY,data TEXT);"
                )
                for item in _label_metadata(path):
                    fragments.execute("INSERT INTO metadata VALUES(?,?)", (str(item["label"]), json.dumps(item)))
                height, width = labels.array.shape
                for y in range(0, height, tile_size):
                    for x in range(0, width, tile_size):
                        check_cancel(cancel)
                        tile = np.asarray(labels.array[y : y + tile_size, x : x + tile_size].compute())
                        if np.issubdtype(tile.dtype, np.signedinteger) and tile.min() < 0:
                            raise ValueError("Negative label values are not supported.")
                        # Rasterio polygonization uses int32, so map IDs locally without precision loss.
                        ids, local = np.unique(tile, return_inverse=True)
                        local = local.reshape(tile.shape).astype(np.int32)
                        for geometry, value in shapes(local, mask=tile != 0, transform=Affine.translation(x, y)):
                            label = str(int(ids[int(value)]))
                            fragments.execute("INSERT INTO pieces VALUES(?,?)", (label, to_wkb(shape(geometry))))
                    fragments.commit()
                    _progress(
                        progress,
                        min(y + tile_size, height),
                        f"Reading label rows {min(y + tile_size, height):,}/{height:,}",
                    )
                batch = []
                for row in fragments.execute("SELECT DISTINCT id FROM pieces"):
                    check_cancel(cancel)
                    label = row[0]
                    # Only one object's fragments are resident, never the full label image.
                    geometry = unary_union(
                        [
                            from_wkb(item[0])
                            for item in fragments.execute("SELECT geometry FROM pieces WHERE id=?", (label,))
                        ]
                    )
                    metadata_row = fragments.execute("SELECT data FROM metadata WHERE id=?", (label,)).fetchone()
                    metadata = json.loads(metadata_row[0]) if metadata_row else {}
                    properties = metadata.get("properties", {})
                    batch.append(
                        AnnotationObject(
                            set_id,
                            metadata.get("id", int(label)),
                            coordinates.transform(geometry, slide.geometry),
                            _class_for(
                                store,
                                properties,
                                classes,
                                set_id,
                                is_ignore=bool(metadata.get("class_is_ignore", False)),
                            ),
                            properties,
                            metadata.get("provenance", {"source": str(path), "label": int(label)}),
                            training=bool(metadata.get("training", True)),
                        )
                    )
                    if len(batch) >= 256:
                        store.append_staged(set_id, batch)
                        batch.clear()
                if batch:
                    store.append_staged(set_id, batch)
            finally:
                fragments.close()
        check_cancel(cancel)
        store.publish_set(set_id)
    except BaseException:
        store.discard_set(set_id)
        raise
    return set_id


def export_labels(
    store: AnnotationStore,
    set_id: str,
    path: str | Path,
    coordinates: CoordinateSpec,
    *,
    mapped_ids: bool = False,
    tile_size: int = 512,
    cancel: Cancel | None = None,
    progress: Progress | None = None,
) -> Path:
    """Write tiled labels and a reversible ID/metadata sidecar, rejecting overlaps."""
    with store.read_snapshot():
        import tifffile
        import zarr
        from rasterio.features import rasterize
        from rasterio.transform import Affine

        if coordinates.units != "px":
            raise ValueError("Raster exports use a pixel grid; physical spacing is written as calibration metadata.")
        path = Path(path)
        sidecar = path.with_name(path.name + ".objects.json")
        if path.exists() or sidecar.exists():
            raise FileExistsError("Choose a new export path; existing image/sidecar pairs are not overwritten.")
        slide = store.slide(store.set_info(set_id)["slide_id"])
        sx, sy = coordinates.downsample
        ox, oy = coordinates.offset
        height = math.ceil((slide.geometry.shape[0] - oy) / sy)
        width = math.ceil((slide.geometry.shape[1] - ox) / sx)
        if min(height, width) <= 0 or min(ox, oy) < 0:
            raise ValueError("Export offset must fall inside the source slide.")
        is_zarr = path.name.endswith(".zarr")
        is_ome = ".ome." in path.name.lower() and not is_zarr
        maximum = np.iinfo(np.uint32).max if is_ome else np.iinfo(np.uint64).max
        classes = {item.id: item for item in store.classes()}
        with TemporaryDirectory(dir=path.parent) as directory:
            root = Path(directory)
            lookup = sqlite3.connect(root / "mapping.sqlite")
            lookup.execute("CREATE TABLE labels(key INTEGER PRIMARY KEY,value TEXT NOT NULL)")
            max_id = 0
            meta_path = root / "objects.json"
            try:
                with meta_path.open("w") as output:
                    metadata = {"frame": coordinates.model_dump(), "slide_geometry": slide.geometry.model_dump()}
                    output.write('{"image2image":')
                    json.dump(metadata, output)
                    output.write(',"objects":[')
                    for index, obj in enumerate(store.iter_objects(set_id=set_id), 1):
                        check_cancel(cancel)
                        value = index if mapped_ids else obj.object_id
                        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
                            raise ValueError(
                                "IDs are not representable in this label format. Enable mapped IDs to export with a reversible sidecar."
                            )
                        max_id = max(max_id, value)
                        lookup.execute("INSERT INTO labels VALUES(?,?)", (obj.key, str(value)))
                        properties = dict(obj.properties)
                        if obj.class_id in classes:
                            item = classes[obj.class_id]
                            properties["classification"] = {"name": item.name, "color": item.color}
                        if index > 1:
                            output.write(",")
                        json.dump(
                            {
                                "label": value,
                                "id": obj.object_id,
                                "properties": properties,
                                "provenance": obj.provenance,
                                "training": obj.training,
                                "class_is_ignore": bool(obj.class_id in classes and classes[obj.class_id].is_ignore),
                            },
                            output,
                        )
                    output.write("]}")
                lookup.commit()
                dtype = np.uint32 if max_id <= np.iinfo(np.uint32).max else np.uint64
                stage = root / path.name
                spacing = (
                    (slide.geometry.spacing[0] * sx, slide.geometry.spacing[1] * sy) if slide.geometry.spacing else None
                )
                if is_zarr:
                    group = zarr.open_group(str(stage), mode="w", zarr_format=2)
                    target = group.create_array(
                        "0", shape=(height, width), chunks=(tile_size, tile_size), dtype=dtype, fill_value=0
                    )
                else:
                    target = None

                def tiles() -> Iterator[np.ndarray]:
                    for y in range(0, height, tile_size):
                        for x in range(0, width, tile_size):
                            check_cancel(cancel)
                            th, tw = min(tile_size, height - y), min(tile_size, width - x)
                            bounds = (ox + x * sx, oy + y * sy, ox + (x + tw) * sx, oy + (y + th) * sy)
                            transform = Affine(sx, 0, bounds[0], 0, sy, bounds[1])
                            tile = np.zeros((tile_size, tile_size), dtype=dtype)
                            for obj in store.iter_objects(set_id=set_id, bounds=bounds):
                                # Reject geometric overlaps, even if too small to affect this output grid.
                                from shapely.geometry import box

                                clipped = obj.geometry.intersection(box(*bounds))
                                for other in store.iter_objects(
                                    set_id=set_id, bounds=obj.geometry.bounds, after=obj.key
                                ):
                                    if clipped.intersection(other.geometry).area > 1e-9:
                                        raise ValueError(
                                            "Objects overlap. Export GeoJSON or a non-overlapping set; labels would overwrite IDs."
                                        )
                                mask = rasterize(
                                    [(obj.geometry, 1)], out_shape=(th, tw), transform=transform, dtype=np.uint8
                                ).astype(bool)
                                if np.any(tile[:th, :tw][mask]):
                                    raise ValueError(
                                        "Rasterized objects overlap; increase export resolution or use GeoJSON."
                                    )
                                value = int(
                                    lookup.execute("SELECT value FROM labels WHERE key=?", (obj.key,)).fetchone()[0]
                                )
                                tile[:th, :tw][mask] = value
                            if target is not None:
                                target[y : y + th, x : x + tw] = tile[:th, :tw]
                            else:
                                yield tile
                        _progress(
                            progress,
                            min(y + tile_size, height),
                            f"Exporting label rows {min(y + tile_size, height):,}/{height:,}",
                        )

                if is_zarr:
                    for _ in tiles():
                        pass
                    unit = {"unit": "micrometer"} if spacing else {}
                    axes = [{"name": axis, "type": "space", **unit} for axis in ("y", "x")]
                    scale = [spacing[1], spacing[0]] if spacing else [sy, sx]
                    origin = (
                        [
                            slide.geometry.origin[1] + oy * slide.geometry.spacing[1],
                            slide.geometry.origin[0] + ox * slide.geometry.spacing[0],
                        ]
                        if slide.geometry.spacing
                        else [oy, ox]
                    )
                    group.attrs.update(
                        {
                            "multiscales": [
                                {
                                    "version": "0.4",
                                    "axes": axes,
                                    "datasets": [
                                        {
                                            "path": "0",
                                            "coordinateTransformations": [
                                                {"type": "scale", "scale": scale},
                                                {"type": "translation", "translation": origin},
                                            ],
                                        }
                                    ],
                                }
                            ],
                            "image-label": {"version": "0.4"},
                            "image2image": metadata,
                        }
                    )
                else:
                    tif_metadata: dict = {"axes": "YX"}
                    if spacing and is_ome:
                        tif_metadata.update(
                            PhysicalSizeX=spacing[0],
                            PhysicalSizeY=spacing[1],
                            PhysicalSizeXUnit="µm",
                            PhysicalSizeYUnit="µm",
                        )
                    kws: dict[str, Any] = (
                        {"resolution": (10000 / spacing[0], 10000 / spacing[1]), "resolutionunit": "CENTIMETER"}
                        if spacing
                        else {}
                    )
                    tifffile.imwrite(
                        stage,
                        data=tiles(),
                        shape=(height, width),
                        dtype=dtype,
                        tile=(tile_size, tile_size),
                        bigtiff=True,
                        photometric="minisblack",
                        ome=is_ome,
                        metadata=tif_metadata,
                        **kws,
                    )
                check_cancel(cancel)
                # Publish the sidecar first; it is removed if publishing the completed image fails.
                os.replace(meta_path, sidecar)
                try:
                    os.replace(stage, path)
                except BaseException:
                    sidecar.unlink(missing_ok=True)
                    raise
            finally:
                lookup.close()
        return path
