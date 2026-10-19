# Annotate

Annotate edits tissue regions and label objects on RGB or multichannel 2D microscopy images. Open **Annotate** in the launcher or **Apps** menu, or run:

```bash
i2i -t annotate
i2i -t annotate -p /path/to/study.i2annotate
```

## First project

1. Click **New project**, choose an empty `.i2annotate` directory, then **Add slides**. Select one slide to work on. Source images remain external; **Relocate** reconnects a moved image.
2. Check **Calibration**. X and Y pixel sizes are independent, in micrometers. An uncalibrated image can still be annotated in pixels. Channel indices in **Annotate** start at zero; choose **RGB** for three-channel color images or **Channels** for independent grayscale channels.
3. Use the shared classes on the left, or **Add class**. Set a color, hide a class, or lock it against edits. Choose an annotation set for the slide.
4. Choose **Polygon** or **Rectangle**, draw, then **Apply drawing**. Double-click finishes a polygon. **Freehand** commits when the drag ends. A brush adds to the selected object; **Brush -** or Alt-drag subtracts and can create holes.
5. **Select** picks an object on the canvas; Shift adds to the selection. Double-click a table row to navigate to it. Search for an exact ID within the active set; `int:12` and `str:12` distinguish typed IDs. The object and proposal tables load 100 rows per page.

**Object properties** shows provenance and lineage and allows editing the original JSON properties. Object IDs remain read-only.

Edits save immediately. **Undo** and **Redo** survive closing and reopening the project. The project contains a versioned manifest, SQLite database with a spatial index and edit journal, model assets, and disposable staging/cache directories. Copy the complete project directory while the editor is closed to make a backup.

## Editing identities

**Vertices** loads only selected geometry. Exterior rings and holes remain separate editable rings; use the brush to add/remove rings or components. Invalid edits remain visible for correction and cannot be committed. They are never silently repaired.

**Cut** draws a line through a selected object. **Apply drawing** previews numbered pieces and lets you choose which retains the original ID; the largest piece is selected by default. Other pieces receive fresh IDs. **Merge** asks which selected ID survives. Both operations retain lineage and are limited to a single annotation set. Disconnected components and holes remain part of the object.

Useful shortcuts:

| Shortcut | Action |
| --- | --- |
| P | Polygon |
| V | Select |
| B | Brush addition |
| Enter | Apply drawing |
| Escape | Clear drawing |
| M | Merge selected objects |
| Delete | Delete selected objects |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo |

## Import and export

**Import** accepts GeoJSON, integer-label TIFF/OME-TIFF, and OME-Zarr directories. Each import gets its own annotation set, so integer `12` and string `"12"` remain distinct, and identical IDs from different files do not collide. Raster zero is background. Fragments sharing a raster value become one object, including disconnected components.

Every exchange asks for an explicit coordinate frame:

- **px**, offset `0, 0`, downsample `1, 1`: full-resolution pixel XY coordinates. The origin is the upper-left pixel corner; pixel centers are `(column + 0.5, row + 0.5)`.
- A pixel patch uses independent X/Y downsampling and a full-resolution XY offset: `slide_xy = offset + patch_xy * downsample`.
- **um** uses `physical_xy = slide_xy * spacing_xy + origin_xy`. Physical exchange requires calibration; patch offsets/downsampling are reserved for pixel exchange.

For [QuPath GeoJSON exchange](https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_annotations.html), use full-resolution pixels and the default offset/downsample. Properties and IDs are retained, with QuPath classification colors and annotation object types. Non-polygonal features must be converted before importing.

**Export** writes accepted objects from the current set. GeoJSON preserves geometry, holes, multipart objects, classes, provenance, training flags, and calibration. Raster export streams tiles and writes an adjacent `.objects.json` sidecar containing original IDs, classes, lineage, calibration, and any ID mapping. Keep this sidecar with its image when reimporting. Select the same explicit offset/downsample used on export; the editor does not guess a coordinate frame.

TIFF and OME-Zarr preserve positive integers through `uint64`; OME-TIFF supports positive IDs through `uint32`. Enable **Map IDs** explicitly for strings, zero/negative IDs, or values outside the destination range. Overlapping objects are rejected rather than overwritten. Raster conversion samples polygons at pixel centers, so arbitrary vector boundaries are quantized to the chosen grid; an integer-label import/export at its original grid is exact.

## Train and predict

1. Draw examples for at least two classes, including background where needed. Assign classes and select **Mark/unmark training examples**. Only explicitly marked accepted annotations are used, across all project slides. Different-class examples must not overlap.
2. In **Train**, set the working downsample, largest feature scale, sample limit per class, and classifier name. Features include multiscale intensity, edges, and texture. Balanced bounded sampling and a fixed random seed make training reproducible.
3. Train, or select a saved classifier in **Predict**. Forest artifacts contain training geometry, settings, class IDs, and library version information. Predictions must use the saved channel indices and resolution.
4. Preview the current view, predict a drawn rectangle, predict a slide, or queue slides. Predictions appear as a separate set in **Review**. Review individual objects or a whole batch, reject them, or **Refine selected** with vertex/brush tools. Acceptance and rejection are undoable.

Predictions record input revision, calibration, channels, preprocessing, model version, and checkpoint hash when a checkpoint is available. Results become stale if their accepted inputs or source image change. A partial review advances the current batch's review revision so the rest can still be reviewed. Recompute stale results.

Processing is local and runs in a spawned worker using the app's Python environment. **Cancel job** stops at the next patch or backend boundary and clears queued work. Imports and predictions remain hidden until complete. On reopening after interruption, incomplete staged sets are discarded. A project lock prevents a second editor from discarding another editor's active work.

## Optional pretrained assistance

Install the desired extra into the environment that runs image2image:

```bash
pip install 'image2image[annotate-sam]'
pip install 'image2image[annotate-cellpose]'
```

- [micro-SAM](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html): choose `micro-sam`, draw foreground/background points or a prompt box, and click **Run point/box assistance**. Keep the prompted region small enough for the displayed working resolution. The default model is `vit_b_lm`.
- [Cellpose](https://cellpose.readthedocs.io/en/latest/cli.html): choose `cellpose`, select up to three image channels and a working resolution appropriate for cells, then predict a view, ROI, or slide. The default model is `cpsam`.

Choose **cpu**, **cuda**, **mps**, or **auto**. Supply a local checkpoint or explicitly enable **Allow model download for this run**. Model downloads are not started automatically on opening the app. Models stay cached in the worker. Overlapping instance patches are reconciled; ambiguous matches are flagged in the review table. Zero detections can be valid and should be assessed at a different resolution or with a suitable model.

This version supports 2D editing and inference. It does not perform 3D editing, deep-model fine-tuning, or spatial-omics table import.

## Performance and validation

Images use lazy source pyramids or lazy display levels for unpyramided sources. Annotation rendering queries visible geometry, uses a class-density overview when a tile contains more than 5,000 objects, and caches 256-pixel RGBA tiles within a 128 MiB limit. The overview is a navigation aid, not an exported segmentation. Full object geometry remains authoritative. One complete selected object must fit in memory, even when it contains many vertices.

Run the reproducible storage/render benchmark:

```bash
python -m image2image.annotation_benchmark --counts 1000000,2000000 --samples 100 --output annotation-benchmark.json
```

The benchmark creates real mixed polygons through the store API, including holes and multipart objects, builds edit history, reopens the database, selects seeded objects, and renders overlays. Each size runs in a separate process. It records every timing sample, peak RSS, cache usage, and database size. It does not include source-image decoding, Qt painting, or model inference. First renders follow import, so operating-system disk caches may already be warm.

Measured on Apple M1 Max, macOS 15.7.3 arm64, 10 logical CPUs, 64 GiB RAM, Python 3.13.9:

| Objects | Selection p95 | Cached overlay p95 | First detailed frame | First overview | Peak RSS |
| --- | --- | --- | --- | --- | --- |
| 1 million | 0.08 ms | 1.40 ms | 1.01 s | 1.76 s | 204.2 MiB |
| 2 million | 0.45 ms | 1.04 ms | 0.95 s | 4.88 s | 180.2 MiB |

Warm selection and cached rendering meet the 100 ms and 250 ms targets for this workload. Cold tiles remain asynchronous and can take seconds. [Raw measurements](../benchmarks/annotation-macos-arm64.json) include the full samples; these are not end-to-end GUI latency claims.

Tests:

```bash
pytest -p no:napari tests/test_annotation.py tests/test_annotation_ml.py tests/test_annotation_qt.py
I2I_ANNOTATION_REAL_MODELS=1 pytest -p no:napari tests/test_annotation_ai_smoke.py
```

The opt-in smoke tests run actual micro-SAM and Cellpose inference on small microscopy images and accept their results through the store. They may download checkpoints and scikit-image sample data. On macOS use a native Qt session; the napari OpenGL canvas does not work reliably with the offscreen Qt plugin.
