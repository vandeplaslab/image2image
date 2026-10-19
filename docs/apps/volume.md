# Volume Builder

Open **Volume Builder** from the launcher or Apps menu, or run `i2i --tool volume`.
It assembles already aligned 2D microscopy slides into one OME-Zarr volume.

1. Drop images onto the window or use **Add images**. For a multi-scene file, choose one scene.
2. Check detected slide numbers on **Slides**. Automatic detection recognizes names such as
   `sample_slide_001.ome.tif` and `sample_section12.czi`. A custom regular expression must contain
   a `section` capture group or one capture group, for example `sample_(?P<section>\d+)_`.
   Edit **Z position** to resolve missing or duplicate numbers. Detection preserves manual edits.
3. Drag rows or use **Move up/down** to reorder images between the existing numbered positions.
   Gaps and excluded slides retain blank Z slices, including excluded endpoints.
4. On **Channels**, select channels, reorder them, and edit their final names. Automatic matching
   ignores case and surrounding whitespace. Use the source mapping selectors below the table to
   resolve aliases or ambiguous names for each slide. Renaming an output does not change its mapping.
5. Choose a missing-channel policy. **Exclude incomplete channels** keeps slides and removes
   selected channels absent from any included slide. **Exclude incomplete slides** retains selected
   channels and leaves blank slices for slides missing any of them. The warning panel and highlighted
   rows explicitly identify exclusions and affected slides. Checkboxes preserve your choices when
   switching policies.
6. On **Output**, choose a directory and a new `.ome.zarr` name, then set the physical section
   spacing in um. Automatic XY spacing uses the coarsest included source pixel size; set a positive
   value to override it. The summary shows dimensions, calibration, dtype, pyramid levels, and the
   estimated uncompressed size. Click **Write OME-Zarr volume**.

Slides share a top-left origin and are resampled with linear interpolation, then zero-padded to
the largest included physical canvas. Included slides must have matching native dtypes. The first
loaded numbered position has physical Z origin zero. Filename numbers and source mappings are
recorded in provenance; missing slices are also marked in the section-presence metadata.

The writer creates OME-Zarr 0.5 / Zarr 3 with automatic XY pyramids and default compression.
It never overwrites an existing output. Cancellation removes the partial store; closing during
export waits for cleanup. The app does not register slides, flatten Z/T stacks, or save editing sessions.
