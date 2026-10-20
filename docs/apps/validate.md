# Registration Validation and Deformation Analysis

Validate compares aligned images or imported registrations using independent landmarks, corresponding region masks, and local deformation
diagnostics. A study can contain several moving images sharing one fixed reference. Each pair keeps separate
annotations, registration candidates, and results. Open **Validate App** from the launcher, or run:

```bash
i2i --tool validate
i2i --tool validate --project_dir comparison.i2validate.json
```

Register's **Tools > Validate registration...** saves a frozen registration snapshot before opening Validate.
Completed Elastix projects have a **Validate...** action in Runner. Validation does not refit registrations or rewrite
their settings. Valis, 3D images, external transform formats, and studies with different fixed references are not supported.

## Set up a comparison

1. Choose fixed and moving images directly to evaluate images already aligned in the same physical coordinate space.
   Add corresponding landmarks and optionally region masks, then analyze; no registration project is required.
   Alternatively, add a Register `.i2r.json`/`.i2r.toml` file or a completed Elastix project directory to evaluate its transform.
   A Register export must contain one fixed image and one moving image. Adding an Elastix project imports every
   compatible registration path. If an empty study encounters several fixed references, choose the shared reference;
   paths to other references, duplicates, and failed imports are listed explicitly.
2. Add other registrations of the same pair. Native image paths, calibration, and reader settings must agree.
   Registrations of another moving image are grouped into a separate pair when they share the fixed reference.
   Use the selector in the **Moving** section to switch pairs, or its **Add moving image...** button to start another pair manually.
   The registration list shows candidates across the whole study. Selecting a candidate switches to its image pair
   and restores that pair's annotations, analysis, and viewer settings.
3. Choose the image channels to display. Fixed and moving viewers are stacked vertically with resizable panels.
   RGB images offer **RGB**, **R**, **G**, and **B** display choices; new RGB imports start in full color.
   Their image toolbars open layer controls for contrast, gamma, opacity, and colormap. Display adjustments do not
   change measurements. Saved registration calibration takes precedence; otherwise confirm the X/Y spacing and origin
   before loading. Calibration source and confirmation are retained in the saved study and reports.
   The edit icon next to each **Choose image...** button updates spacing and origin. Existing annotations stay attached
   to the same native pixels. Fixed calibration changes apply to every pair; imported transforms are expressed in the
   updated coordinate system. The delete icon removes an image from the app without deleting source files. Removing
   the fixed reference clears dependent registration candidates and landmarks across all pairs.
4. In **Analysis**, the requested analysis downsample applies to the selected pair's diagnostic images, masks, and deformation maps;
   a four-million-pixel cap limits the common fixed-image grid. Landmark calculations use native physical coordinates.
   Native RGB/channel display is independent of the saved scalar analysis channel used for magenta/green overlap
   diagnostics and exported figures. Changing native display does not invalidate numerical results.
5. Save the study as `.i2validate.json` to retain annotations, exclusions, mask geometry, candidates, and settings.
   Existing version 1 pair sessions open as a one-pair study; new saves use version 2.
   Changed or missing inputs are reported when reopening. Re-add a changed candidate to accept its new definition.

## Independent landmarks

In **Landmarks**, choose **Add pair**, click the fixed image, then the corresponding location in the moving image.
Enable **Edit landmarks in the viewers** to drag existing markers, or edit coordinates in the table. The image layer
is selected for image controls by default, and is restored after adding a pair. Selecting a row focuses its location.
**Link physical view centers and zoom** keeps both viewers on the same physical region and magnification. Disable it
for independent navigation while annotating images in different native frames. Uncheck **Use** to exclude a pair;
use **Note** to explain the exclusion.
Native viewers use lazy image pyramids, retaining the full-resolution level for zooming. Large images without stored
pyramids receive lazy display levels. The analysis downsample does not reduce native viewer detail. Coordinates,
image layers, landmarks, and scalebars use `um`, including images with different X/Y spacing and nonzero origins.

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
For image-only comparisons, `G` is the identity mapping: residuals are the direct differences between the calibrated
landmark coordinates. Validation landmarks are not used to fit a transform. All distance summaries, region overlap,
coverage and identity-map diagnostics can be calculated directly. Original fitting residuals and historical registration
deformation are unavailable without the original registration; an identity Jacobian is not evidence of no past deformation.
Each row includes X/Y residuals and Euclidean distance. Summaries include count, mean, median, RMS, 95th percentile,
and maximum. The evaluated count and eligible count remain visible when a transform cannot evaluate every landmark.

Known fitting points are never automatically copied into validation. Reuse of a fitting landmark from any candidate
excludes that landmark from the shared validation set. Otherwise, independence is user-declared: the application cannot
know whether an externally supplied point was previously used to tune a registration. Landmark errors describe sampled
locations and are not interpolated into a claim of accuracy across unmeasured tissue.

The **Validation errors** view includes a visible legend and summary beside the image. Point colors encode positional
error in `um`, with a numeric colorbar shared across registration candidates for the current pair. Cyan arrows run
from observed moving landmarks to their predicted positions at actual physical length. Optional point labels show
each landmark ID and error. The native moving image provides context; its intensity is not an error measurement.
The legend explains missing or excluded evidence, and zero error remains distinct from an unavailable measurement.

When switching to a pair without landmarks, Validate offers to copy existing validation pairs from another moving
image. The previous pair is selected as the source when available; another annotated source can be chosen.
Copying preserves physical coordinates, notes, and **Use** flags, with independent editable rows in the destination.
Check that the copied moving coordinates still identify the correct tissue locations. Existing bounds and known-fitting
exclusions continue to apply. Declining leaves the destination empty, and existing annotations are never overwritten.

Register's existing aggregate value is labeled **Fitting error - root-sum-square**. Its numerical definition is unchanged.
It grows with the number of fitted landmarks and is not independent validation accuracy. Validate reports fitting
statistics separately, in its common moving physical frame. Elastix projects without original fitting landmarks show
that evidence as unavailable.

## Corresponding masks

The **Region overlap (optional)** tab evaluates how well corresponding anatomical segmentations overlap after
registration. It is optional evidence alongside landmarks, rather than a required registration step.
Load a fixed and moving binary or integer-label mask. Confirm each mask's pixel spacing and origin in `um`.
Masks matching the native image shape default to that image's geometry. Select corresponding anatomical labels in the
table; matching positive IDs detected in a sampled preview are proposed automatically. Add small regions or unmatched IDs
explicitly. Label zero is background. This version loads masks; it does not create segmentations.
Mask overlays appear on the native viewers while this tab is selected. A missing second mask disables overlap scores
while allowing landmark, fitting, and deformation measurements to run.

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

## Analysis and export

**Analyze all**, at the top of **Analysis**, evaluates pairs sequentially in the background and records pair failures
separately. Use **Cancel** beside it to stop a run. Editing evidence invalidates that
pair's results; results for other pairs remain available. Switching moving images restores their annotations and results.

**Analysis** separates independent landmark accuracy, original fitting residuals, optional region overlap, and deformation
statistics. Unavailable measurements explain the evidence needed. Registrations without validation landmarks can still
show fitting statistics and deformation; fitting residuals are not independent accuracy. Use the candidate and map
selectors to inspect image overlap, point residuals, mask disagreements, signed Jacobians, and area change.
Residual distances use the moving physical frame in `um`. No universal pass/fail threshold is applied.
Help bubbles beside the map selector, measurement tables, and downsample control explain interpretation and unavailable evidence.

**Export report...**, at the bottom of **Analysis**, creates a new directory containing:

- A self-contained `report.html`, with embedded diagnostic images, metric tables, exclusions, units, and explanations.
- Combined `summary.csv`, `landmarks.csv`, and `masks.csv`, with pair identifiers.
- `results.json` and a copy of the versioned validation study, including pair failures.
- A subdirectory per analyzed pair containing its full report, candidate PNG figures, and compressed NPZ numerical maps.

Measurements remain separate for each image pair; errors are not pooled across moving images.

NPZ arrays use YX indexing. `spacing_xy_um` and `origin_xy_um` describe their physical grid. `valid`, `coverage`, and
`nonpositive` distinguish usable derivatives, source coverage, and nonpositive determinants. Numerical values remain
available independently of the display colors. Transform definitions are fingerprinted by content; large image inputs
are fingerprinted by file size and modification time. Reports also retain image geometry and software versions.
