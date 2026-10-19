# Illumination and Background Correction

Open **Illumination Correction** from the launcher or Apps menu, or run:

```sh
i2i --tool illumination --file image.ome.tiff
```

The app corrects 2D grayscale, multichannel, RGB, and whole-slide microscopy images.
TIFF/CZI scenes are selected explicitly. OME-Zarr image groups are supported, including
OME-NGFF 0.4 and 0.5 metadata. Z stacks, time series, RGBA, and 64-bit integer inputs
are rejected rather than silently reduced. RGB exports contain three named R/G/B channels.

## Workflow

1. Add images on **Images**, or drop files onto the window. Use **Add OME-Zarr** for image
   directories. Choose a scene for every multi-scene file. Check the images to export.
2. Select a row and preview channel. That channel selection affects the preview only;
   every source channel is exported. The shared recipe requires matching channel names
   and order. Use **Use selected image's channel layout** to explicitly change that layout.
3. On **Correction**, optionally load one prepared master dark frame and one master flat
   frame. Select their scenes and check the channel mappings. Unique matching names are
   suggested; sharing one reference channel across multiple source channels requires
   selecting it in each mapping. References must match source dimensions and known pixel
   sizes. The app does not resize, register, average, or exposure-scale calibration frames.
4. Choose a background estimator and adjust its parameters. **Overview** samples native
   pixels on a reduced grid and is approximate. Double-click the original image, or set
   X/Y and size, to compute an exact **Full-resolution region** with its surrounding context.
5. Compare the linked original and result views. Switch the result to **Estimated
   background** to inspect the field being subtracted. Display limits are shared by default;
   **Fit both** includes both intensity ranges. These controls never change exported pixels.
6. On **Export**, choose a directory, format, and output dtype, then export the checked
   images. The running batch uses frozen settings. Incompatible or failed images are
   reported individually; other valid images continue. Cancellation retains completed
   images and removes the current partial output.

Use **File > Save recipe** and **Load recipe** to reuse settings. Recipes include reference
paths, scenes, file identities, and channel mappings. Changed reference files must be reloaded.

## Correction and intensity handling

Each channel is processed independently, in this order:

- Dark subtraction: `A = image - dark`. Disabled dark correction uses zero.
- Flat correction: `G = flat - dark`, then `A = A * mean(G) / G`. The mean comes from
  the complete reference channel, not the preview region or individual export tiles.
  Select **Flat already dark-corrected** to use `G = flat` without subtracting dark twice.
- Background subtraction: estimate `B` from the calibrated image, then calculate `A - B`.

The estimators are:

- **Gaussian:** smooth the calibrated image using sigma in native pixels, a support radius
  of `ceil(4 * sigma)`, and reflected boundaries. Foreground structures also influence this
  estimate; use a scale larger than the features of interest.
- **Morphology:** opening for bright features or closing for dark features, using a square
  footprint of side `2 * radius + 1` and reflected boundaries.
- **Rolling ball:** an ellipsoidal rolling-ball estimate with independently specified spatial
  radius and intensity height. Height remains in the image's intensity units; input values
  are never normalized to 0-1. Dark-feature estimation applies the estimator to the negated
  image, then restores the sign. Large radii can be expensive.

Dark-feature subtraction produces signed negative residuals; it does not invert the image
or restore a white background. All correction stages are optional; the initial recipe is
unchanged intensities with float32 output.

Calculations use float64 intermediates. **Float32**, the default export type, retains
negative values without rescaling. **Source**, **uint8**, and **uint16** explicitly select
storage types. Integer output clips to the type's limits, then rounds to nearest even.
Float output that exceeds its finite range is rejected. The corrected preview shows the
selected export conversion, while its statistics describe values before conversion and
predict the number of clipped pixels. Overview statistics cover sampled pixels only.

Non-finite inputs and invalid flat responses block the affected image. A normalized flat
response at or below **Minimum normalized flat** (default `0.000001`) is invalid; the app
reports the invalid-pixel count rather than replacing it with a small denominator.

## Outputs and reproducibility

New images are named `NAME.scene-N.corrected.ome.tiff` or
`NAME.scene-N.corrected.ome.zarr`. Existing images and parameter records are never overwritten.

OME-TIFF uses BigTIFF, lossless Deflate compression, and tiled pyramids. OME-Zarr uses
OME-NGFF 0.4 metadata with Zarr 2 storage for compatibility with the suite's 2D reader.
Pyramid levels use 2x2 means, edge padding for odd dimensions, and nearest-even integer
rounding. Base-resolution corrected pixels are independent of the pyramid.

Both formats retain channel names/order and available separate X/Y calibration. Unknown
pixel sizes remain uncalibrated. TIFF export stages a chunked image on disk, so sufficient
free space for that working image and the final TIFF is required.

Each output has embedded correction metadata and a matching `.json` sidecar containing
source and calibration paths, size/timestamp identities, scene/channel selections, the
complete recipe, flat normalization means, output dtype, software versions, and measured
per-channel ranges, negative counts, and clipping counts. No changes are written into the
source image. Recipes do not substitute for keeping the original image and calibration files.

Native-region previews and export share the same tiled correction code. Filter context is
included around each tile to avoid seams. Neighborhoods above the processing memory limit
are rejected with a request to reduce the radius or region size. Cancellation is checked
between tiles and filtering steps; an active decoder/filter call must finish before closing.
