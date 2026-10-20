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

The collapsible **Overview** sits below the slide controls; click its image to navigate. Drag in **Select** to pan and scroll to zoom. Freehand and brush gestures save one undoable edit on release without moving the canvas. The brush cursor shows its radius, and a temporary outline shows the painted footprint while dragging: green for addition and red for subtraction. Escape cancels an unfinished stroke.

The Annotate and Review tables show their page number, displayed/total object counts, and the 100-object page size. Each table keeps its own page position; Previous and Next are disabled at the boundaries.

New projects include Tissue, Stroma, Background (amber), Nucleus, Cells, Ignore, Other, and Necrosis. Existing projects keep their palettes. **Edit class** includes **Use as Ignore mask**; this behavior stays with the class if it is renamed.

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

1. Draw examples for at least two classes, including Background where needed. New manual drawings and imports are included automatically. The **Training** checkbox in Annotate controls inclusion; **Include in training** and **Exclude from training** update selected objects together. Only accepted, classed, non-Ignore annotations contribute, across all project slides. Existing projects and explicit imported flags retain their saved choices. Reviewed predictions stay excluded until checked.
2. In **Train classifier**, inspect counts by slide and class. These count candidate objects before masking and sampling. Accepted Ignore regions exclude overlapping pixels from training examples in the same annotation set, regardless of their Training checkbox. Different-class examples must not overlap after masking. Fully masked classes or classes without pixels at the working resolution produce an explanatory error.
3. Set the classifier name, working downsample, largest feature scale, sample limit per class, random seed, and maximum tree depth. Training starts when you click **Train from included annotations**. The model is a CPU random forest with 100 trees. Seed 42 and depth 24 are the defaults; use the same seed and inputs for reproducible training. Greater depth can fit finer detail and increase overfitting. Features include multiscale intensity, edges, and texture. A downsample of 4 uses one working pixel per 4 x 4 source pixels; feature scales are in working pixels. Sampling balances classes to the smallest available count, up to the chosen limit.
4. In **Predict**, use `forest` and select a classifier from the project's `models` directory. The library refreshes after opening, training, or loading a model. **Load classifier...** validates and copies an external `.npz` file into that directory, preserving existing files on name collisions. Class IDs and library versions must be compatible. Its name, classes, and actual parameters are displayed. Saved channels and resolution are used automatically, independently of the current display channels. Model artifacts retain training geometry, Ignore masks, settings, class IDs, and library versions. Older artifacts without a depth setting use 24.
5. Preview the current view or predict a drawn rectangle. The attached rectangle button selects the Rectangle drawing tool. Full-slide and queued-slide actions sit below the separator. Predictions appear as a separate set in **Review**, with class, area, and review/seam warnings. Select a proposal to inspect its source backend and model. Accept or reject individual objects or a whole batch, or **Refine selected** with vertex/brush tools. Acceptance and rejection are undoable.

Predictions record input revision, calibration, channels, preprocessing, model version, and checkpoint hash when a checkpoint is available. Results become stale if their accepted inputs or source image change. A partial review advances the current batch's review revision so the rest can still be reviewed. Recompute stale results.

Processing is local and runs in a spawned worker using the app's Python environment. **Cancel job** stops at the next patch or backend boundary and clears queued work. Imports and predictions remain hidden until complete. On reopening after interruption, incomplete staged sets are discarded. A project lock prevents a second editor from discarding another editor's active work.

Open **Task queue** in Predict or the queue icon in the toolbar to see the active phase and waiting tasks, including slide names, backend, and model. Tasks run in order; cancelling clears the waiting tasks too.

Predict shows the active processing phase, progress, elapsed time, and an approximate ETA when enough patches have completed. The estimate covers the active phase; model loading, single prompted inference, and unmeasured postprocessing show indeterminate progress. Queued slides also show the current slide number. ETA is based on measured work and can change with image content and hardware.

## Optional pretrained assistance

Install the desired extra into the environment that runs image2image:

```bash
pip install 'image2image[annotate-sam]'
pip install 'image2image[annotate-cellpose]'
```

micro-SAM and Cellpose use independent pretrained models; **Train classifier** does not fine-tune them. Availability and workflow guidance appear immediately below **Backend**. Unavailable or incompatible actions are disabled. Each backend retains its model settings when you switch away and back.

Select a model from the installed backend's catalogue, then click the download icon directly beside the selector. Hover over the help icon for the model's purpose, encoder size, and requirements. Descriptions follow the [micro-SAM model guide](https://computational-cell-analytics.github.io/micro-sam/) and [Cellpose model guide](https://cellpose.readthedocs.io/en/latest/models.html). Model entries also have descriptive tooltips. Downloads can run before opening a project. The status beside the selector indicates whether weights are cached; predictions reuse complete cached weights without another download checkbox. Opening the app or selecting an entry never starts a download. The app does not install backend packages automatically. Models requiring additional packages, such as MobileSAM or DINO, remain disabled with an explanation.

Managed weights live under `USER_CACHE_DIR / "models" / backend`, using the platform-specific image2image cache directory defined in `_appdirs.py`. Hover over the model status to see the directory. SAM downloads include the encoder and any registered decoder. Known checksums are verified; incomplete or cancelled downloads are not marked ready. During a download, the download icon becomes a stop button. Use it, **Cancel job**, or the queue's cancel button to stop the download, including a stalled connection. Partial files owned by that download are removed; complete verified assets can be reused on retry. A local checkpoint can also be selected explicitly.

- [micro-SAM](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html): choose `micro-sam` for automatic view, ROI, or slide prediction. It uses decoder-based instance segmentation when decoder weights are available, otherwise automatic mask generation with library defaults. Both run in overlapping patches and send instances to Review. For prompted assistance, use the attached **Prompt +**, **Prompt -**, and **Prompt box** actions to select drawing tools, mark several regions, then click **Run point/box assistance**. Each positive point and each box produces its own instance; negative points are shared exclusion hints and do not start an instance. Boxes remain visible while switching between prompt tools. Escape clears all prompts. Keep the prompted region small enough for the working resolution. The default model is `vit_b_lm`.
- [Cellpose](https://cellpose.readthedocs.io/en/latest/cli.html): choose `cellpose`, select up to three image channels and a working resolution appropriate for cells, then predict a view, ROI, or slide. The default model is `cpsam`.

For pretrained models, choose **cpu**, **cuda**, **mps**, or **auto**. Auto selects CUDA, then Apple MPS, then CPU, depending on availability. These settings do not affect forest training or prediction, which use CPU. SAM and Cellpose instances receive the selected shared class; they do not use the forest classifier to classify detections. Models stay loaded in the worker when possible. Overlapping instance patches are reconciled; ambiguous matches are flagged in the review table. Zero detections can be valid and should be assessed at a different resolution or with a suitable model.

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
pytest -p no:napari tests/test_annotation.py tests/test_annotation_ml.py tests/test_annotation_models.py tests/test_annotation_qt.py
I2I_ANNOTATION_REAL_MODELS=1 pytest -p no:napari tests/test_annotation_ai_smoke.py
```

The opt-in smoke tests run actual micro-SAM and Cellpose inference on small microscopy images and accept their results through the store. They may download checkpoints and scikit-image sample data. On macOS use a native Qt session; the napari OpenGL canvas does not work reliably with the offscreen Qt plugin.
