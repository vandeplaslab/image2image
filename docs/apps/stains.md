# Stain Separation and Normalization

Open **Stain Separation and Normalization** in the launcher, **Open Stain Separation** in
the Apps menu, or run:

```sh
i2i --tool stains --file slide.ome.tiff
i2i --tool stains --file_dir /path/to/slides
```

The app separates two or three brightfield stains and optionally reconstructs their
appearance using a reference slide. H&E and H-DAB presets are included, alongside custom
two- and three-stain profiles. It accepts 2D uint8/uint16 RGB images through i2i's existing
readers, including supported whole-slide TIFF formats, CZI, and OME-Zarr image groups.
Z/time stacks, alpha channels, and other pixel types are rejected explicitly.

## Workflow

1. **Images:** add or drop images, choose scenes for multi-scene files, and check the
   images to export. Select a row to inspect it. For unlabeled three-channel data, map
   the channels to R, G, and B explicitly in **Stains and Reference**.
2. **Stains and Reference:** select a stain type. **Auto per slide** estimates two vectors
   independently for each source. **Shared vectors** applies one source profile across
   the batch. Custom three-stain profiles use manual vectors; automatic estimation is
   available for two stains only.
3. Inspect the linked original, stain channels, and residual views. **Stain colors** and
   **Display OD max** only change the display. The residual is signed and shows color
   information outside the two-stain plane.
4. To refine a profile, choose **Selected source image** or **Reference image** in the
   editor. Edit vector components and click **Apply vector edits**, or **Draw sampling
   rectangle** and then estimate both stains, sample one pure stain, or sample blank
   background. Vectors are normalized automatically. Background values are fractions
   of the source's dtype maximum, so white is initially `(1, 1, 1)`.
5. Manual source edits override automatic/shared settings for that image. **Reset image
   vectors** removes that override. **Use these source vectors for the batch** explicitly
   replaces all source overrides with the editor profile. Custom stain names are shared
   across all profiles; colors remain independently editable.
6. Optionally load a reference, or use the selected image as the reference. Source and
   reference must represent the same stains in the same order, but may have different
   dimensions and pixel sizes. **Reference normalization** shows the normalized source
   alongside an independently navigable reference. Color matching is the default;
   intensity matching is an explicit additional option.
7. **Export:** choose stain channels, dtype, format, and output directory. Adding a
   reference enables normalized RGB export; this can be unchecked independently.
   Export the checked images. Completed images remain available if later images fail
   or the batch is cancelled. Failures are listed individually.

Double-click the original to center a native-resolution crop, or set X, Y, and size.
Existing image pyramids provide overview navigation. Without a suitable pyramid, the
preview shows a bounded center crop; its native origin is displayed beneath the views.
Native crops and full exports share the same transform. Reduced pyramid previews are
approximate because averaging RGB pixels changes their optical densities.

If automatic estimation fails, the original remains visible for manual region selection.
Choose a representative region with both stains, or use a preset/manual profile.
Estimation never silently falls back to another profile during export.

Use **File > Save recipe** and **Load recipe** to retain the reference, its file identity,
profiles, sampling settings, and source overrides. A sampled profile is saved as resolved
vectors; the temporary rectangle itself is not needed to reproduce the transform.

## Numerical behavior

Separation uses natural-log optical density:

```text
OD = -ln(min(max(pixel, 1) / background_native, 1))
coefficients = OD @ inverse(stain_matrix)
```

Stain vectors are unit row vectors. Two-stain profiles receive an orthogonal third
vector; its coefficient is the residual. The three coefficients remain signed.
Zero pixels are clamped to one native intensity unit before taking logarithms. Input
values above the selected white value have zero OD in that component.

Two-stain estimation uses a Macenko-style covariance plane and angular percentiles,
with directions matched to the starting profile to maintain stain order. It samples
at most 100,000 native pixels from at most 64 stratified 256-pixel patches. White,
very dense, and clipped pixels are excluded using the visible advanced thresholds.
This is a bounded sample, not an exhaustive tissue census; small tissue areas can
require an explicitly selected region.

Color-only normalization retains source stain coefficients and uses reference vectors
and background values for RGB reconstruction. Intensity matching additionally multiplies
each real stain coefficient by the reference/source positive-coefficient percentile ratio
(99th percentile by default). The source residual contribution is retained. Missing stain
signal prevents intensity matching. Reconstructed RGB is clipped to its representable
range and rounded to the source bit depth.

Raw stain channels always describe the original slide and are independent of reference
normalization. These are optical-model coefficients, not absolute stain concentrations.
The approach and practical limitations are described in the
[QuPath stain separation tutorial](https://qupath.readthedocs.io/en/stable/docs/tutorials/separating_stains.html)
and the [Macenko paper](https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf).

## Outputs and performance

Outputs use `NAME.scene-N.stains.ome.tiff` and, when selected,
`NAME.scene-N.normalized.ome.tiff`; OME-Zarr uses the corresponding `.ome.zarr` suffix.
Existing images and parameter records are never overwritten. Selected stain channels
are stored together with their names. Normalized TIFFs are RGB images; normalized Zarr
stores contain named R/G/B channels.

- **Float32**, the default stain output, preserves signed coefficients without rescaling.
- **Uint16** registration output uses one frozen positive-coefficient percentile per
  channel and slide. It clips negatives to zero, maps that percentile to 65535, clips
  larger values, and rounds. The scale is recorded; it never depends on tile contents.

Outputs preserve native dimensions and known separate X/Y calibration. Unknown spacing
remains uncalibrated. Both formats have lossless pyramids using 2x2 means, edge padding
for odd sizes, and rounding for integer levels. OME-Zarr uses NGFF 0.4 metadata with Zarr
2 storage for compatibility with the existing 2D readers.

Each image includes provenance and a matching JSON sidecar: source identity, recipe,
resolved source/reference vectors, sampling counts, intensity scales, channel names,
negative coefficient counts, and software version. Batch settings are frozen at export.

Processing uses NumPy fitting, cached Numba kernels, optical-density lookup tables,
and 512-pixel output tiles. RGB components are decoded together. TIFF/Zarr reads remain
lazy; CZI regions read intersecting native subblocks instead of converting the whole
slide first. The CZI decoded-block cache is bounded at 32 MiB; files with individual
subblocks above 128 MiB require conversion to a tiled format. Mosaic gaps retain the
existing reader's zero fill. Decoder memory also depends on the source chunk/strip size.

Exports stage a chunked pyramid on disk before publication. TIFF therefore needs space
for the working pyramid and final output. Cancellation is checked between reads, tiles,
and pyramid steps; an active decoder call must finish before cleanup.

Run the reproducible benchmark with:

```sh
python -m image2image.stains_benchmark --size 4096 --repeats 3 --output stains-benchmark.json
```

It reports first-call Numba time separately from warmed kernel throughput, fitting time,
first/warm preview latency, complete export time including pyramids and compression,
and sampled process RSS. Fixtures are noisy synthetic mixtures with spatially varying
tissue/background, generated outside the export timing. For a true compilation-cold run,
set `NUMBA_CACHE_DIR` to a fresh temporary directory before launching the process.
Synthetic throughput does not measure every scanner codec or guarantee stain quality on
real tissue; inspect the separated channels and residual on representative slides.
