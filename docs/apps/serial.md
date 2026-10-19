# Serial Sections app

Open **Serial Sections** from the launcher, or run `image2image -t serial`.
The original `image2image -t elastix3d` preparation modes remain available.

## Prepare sections

1. In **Sections**, load one RGB or multichannel image per physical section.
2. Double-click the **section** column to edit physical section numbers. New images
   receive consecutive numbers in natural filename order. Sorting the table does
   not change physical order.
3. Uncheck **keep** to exclude a section while retaining its empty position.
   For example, numbers 3, 5 and 8 produce six Z positions, including gaps at 4,
   6 and 7. Excluding section 3 keeps the first position empty.
4. Use the existing rotation, flip and translation controls to prepare orientation.
5. In **Registration**, set positive section spacing in micrometers and choose an
   included reference. The initial reference is the middle included section.

Section numbers must be unique positive integers. Z=0 is the lowest number in
the project, including excluded sections. All included images must have matching
channel names/order, RGB layout and intensity dtype. XY calibration comes from
the existing image reader; check the source image's physical pixel size.

Save with the existing project controls. Serial settings are optional additions
to `.i2wsiprep.json` or `.i2wsiprep.toml`; older projects still load.

## Register

Choose **Rigid**, **Rigid + affine**, or **Rigid + affine + deformable**. Rigid is
the default. Each included section registers to its nearest included neighbor
toward the reference, and transformations are composed along that path.

Choose the registration channel independently of the exported channels.
RGB images use the existing brightfield preprocessing. Registration downsampling
and the pixel cap affect alignment processing only. Existing Elastix presets are
used, with coarse levels limited on small images to avoid degenerate pyramids.
For a shortened deformable pyramid, the optimizer uses Elastix's default step
size based on voxel spacing, avoiding the larger steps intended for whole slides.
See the [Elastix optimizer parameters](https://elastix.dev/doxygen/parameter.html).

**Register sections** saves a project snapshot and runs the headless command
through the subprocess queue. Progress and failures appear in the workflow tabs.
**Cancel processing** cancels this window's job. Input, inclusion, orientation,
channel or registration-setting changes make earlier registration results stale.
Run snapshots and registration logs remain available for inspection.

## Reconstruct and inspect

In **Volume**, the initial canvas is the prepared reference image's physical XY extent.
Edit X, Y, width and height to expand or crop it. Output XY spacing defaults to
the coarsest native input spacing. **Use reference canvas** restores these defaults.

**Reconstruct and export** writes every source channel using the same composed
transformation, with linear interpolation and the source intensity dtype. Missing
or excluded sections stay empty; no Z interpolation is performed. Display contrast
and registration intensity preprocessing do not change quantitative output values.

Export produces a Zarr 3 store with OME-Zarr 0.5 metadata and CZYX arrays. XY
pyramids use 2x2 block means while keeping Z sampling unchanged. Calibration,
channel names, section presence and reconstruction provenance are stored alongside
the arrays. Output is written in tiles; source regions larger than 64 MiB,
including the SimpleITK copy, are subdivided before resampling. Storage codecs may
also buffer their native source chunks. Reconstruction keeps only one section's
reader and composed transform open at a time.

The store is published only when all sections and pyramid levels succeed. Choose
a new output name for each reconstruction. Existing completed stores are preserved
on cancellation or failure. Forced termination can leave a hidden `.partial`
directory; it is not a completed result and can be removed after the job stops.

Completed results open in linked XY/XZ/YZ views with micrometer readouts, channel
selection and shared display contrast controls. Click a slice or move its slider to move the linked
position. The 3D view is explicitly downsampled and limits its preview array to
64 MiB. **Open volume** reopens a completed store without requiring source sections.

## Headless commands

```sh
image2image -t serial -p stack.i2wsiprep.json
image2image serial register --project stack.i2wsiprep.json
image2image serial reconstruct --project stack.i2wsiprep.json --output stack.ome.zarr
```

The commands update completed-result locations in the supplied project. The GUI
uses a separate snapshot per run and adopts results only when the live project
still matches that run.
