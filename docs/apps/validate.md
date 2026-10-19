# Registration Validation and Deformation Analysis

Validate compares registrations of one fixed/moving image pair using independent landmarks, corresponding label masks,
and local deformation diagnostics. Open **Validate App** from the launcher, or run:

```bash
i2i --tool validate
i2i --tool validate --project_dir comparison.i2validate.json
```

Register's **Tools > Validate registration...** saves a frozen registration snapshot before opening Validate.
Completed Elastix projects have a **Validate...** action in Runner. Validation does not refit registrations or rewrite
their settings. Valis, 3D images, external transform formats, and comparisons across different image pairs are not supported.

## Set up a comparison

1. Add a Register `.i2r.json`/`.i2r.toml` file or a completed Elastix project directory. A Register export must contain
   one fixed image and one moving image. For an Elastix project with several registration paths, select the image pair.
2. Add other registrations of the same pair. Native image paths, calibration, and reader settings must agree.
3. Choose the image channels to display. The analysis downsample applies to images, masks, and deformation maps;
   a four-million-pixel cap limits the common fixed-image grid. Landmark calculations use native physical coordinates.
4. Save the session as `.i2validate.json` to retain annotations, exclusions, mask geometry, candidates, and settings.
   Changed or missing inputs are reported when reopening. Re-add a changed candidate to accept its new definition.

## Independent landmarks

In **Landmarks**, choose **Add pair**, click the fixed image, then the corresponding location in the moving image.
Drag existing markers or edit their coordinates in the table. Selecting a row centers both views on that pair.
Zoom is synchronized in physical units. Uncheck **Use** to exclude a pair; use **Note** to explain the exclusion.
Image previews are capped at four million pixels using bounded source reads. Their layer names show the displayed
resolution; annotation coordinates and landmark calculations retain physical precision independently of this preview.

CSV imports use these columns:

```csv
id,fixed_x,fixed_y,moving_x,moving_y
landmark-1,120,240,122,238
landmark-2,310,170,307,174
```

Select `px` for coordinates in each image's native pixel grid, or `um` for physical coordinates. X is the column axis,
Y is the row axis, and pixel coordinate zero is the center of the first pixel. The table and views always use `um`.
CSV export includes all coordinates; exclusions and notes are retained in the session and full report.

Validation evaluates the fixed-to-moving map `G` and reports `G(fixed) - moving` in the **moving physical frame**.
Each row includes X/Y residuals and Euclidean distance. Summaries include count, mean, median, RMS, 95th percentile,
and maximum. The evaluated count and eligible count remain visible when a transform cannot evaluate every landmark.

Known fitting points are never automatically copied into validation. Reuse of a fitting landmark from any candidate
excludes that landmark from the shared validation set. Otherwise, independence is user-declared: the application cannot
know whether an externally supplied point was previously used to tune a registration. Landmark errors describe sampled
locations and are not interpolated into a claim of accuracy across unmeasured tissue.

Register's existing aggregate value is labeled **Fitting error - root-sum-square**. Its numerical definition is unchanged.
It grows with the number of fitted landmarks and is not independent validation accuracy. Validate reports fitting
statistics separately, in its common moving physical frame. Elastix projects without original fitting landmarks show
that evidence as unavailable.

## Corresponding masks

Load a fixed and moving binary or integer-label mask in **Masks**. Confirm each mask's pixel spacing and origin in `um`.
Masks matching the native image shape default to that image's geometry. Select corresponding anatomical labels in the
table; matching positive IDs detected in a sampled preview are proposed automatically. Add small regions or unmatched IDs
explicitly. Label zero is background. This version loads masks; it does not create segmentations.

The moving mask is sampled onto the shared fixed grid using nearest-neighbor interpolation. Dice, IoU, and areas are
reported per label correspondence. Both-empty regions are unavailable; one-empty regions score zero. A coarse grid can
miss small regions. Moving content outside the fixed analysis field is clipped, and source-field coverage is reported.
The fixed, moving, and joint mask coverage fractions identify incomplete mask fields of view.
Choose a finer analysis grid when evaluating small structures.

## Deformation maps

The signed Jacobian is the determinant of the derivative of the complete **fixed-to-moving** map in physical units.
For positive determinants, the **moving-to-fixed area ratio** is its reciprocal:

- Ratio 1, or log2 ratio 0: unchanged local area.
- Positive log2 ratio: local expansion when mapping moving tissue into fixed space.
- Negative log2 ratio: local compression in that direction.
- Nonpositive Jacobian: orientation reversal or singularity. This can include intentional reflections and is not
  automatically classified as tissue folding.

Maps include global affine scaling, not just the nonlinear residual. Derivative boundaries, non-finite results, and
regions without source coverage are unavailable. Sampling resolution limits the spatial detail that can be diagnosed.
These maps describe the registration deformation and do not by themselves establish biological deformation.

## Results and export

**Analyze** runs in the background and can be cancelled. Editing inputs invalidates existing results. Use **Results**
to switch candidates and inspect overlap, point residuals, mask disagreements, signed Jacobians, and area change.
No universal pass/fail threshold is applied.

**Export report...** creates a new directory containing:

- A self-contained `report.html`, with embedded diagnostic images, metric tables, exclusions, units, and explanations.
- `summary.csv`, `landmarks.csv`, and `masks.csv`.
- `results.json` and a copy of the validation session.
- One PNG diagnostic figure and compressed NPZ numerical maps per candidate.

NPZ arrays use YX indexing. `spacing_xy_um` and `origin_xy_um` describe their physical grid. `valid`, `coverage`, and
`nonpositive` distinguish usable derivatives, source coverage, and nonpositive determinants. Numerical values remain
available independently of the display colors. Transform definitions are fingerprinted by content; large image inputs
are fingerprinted by file size and modification time. Reports also retain image geometry and software versions.
