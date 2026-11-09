"""Constant values."""

from image2image.config import STATE

UINT8_TOOLTIP = (
    "Convert to uint8 to reduce file size with minimal data loss. This will result in change of the"
    " dynamic range of the image to between 0-255."
)
UINT8_WARNING = (
    "While this option reduces the amount of space an image takes on your disk, it can lead to data loss<br>"
    " and should be used with caution. If you are exporting RGB images (e.g. H&E or PAS), then this is not<br>"
    " a concert since that data is already in the 0-255 range. However, if you are exporting images with a<br>"
    " dynamic range greater than 0-255, then you should be careful (e.g. AF, IHC, CODEX)."
)

REGISTRATION_PATH_HELP = (
    "Type or registration path. There are several types including:<br>"
    "<b>rigid</b> - Translation and rotation<br>"
    "<b>similarity</b> - Translation, rotation, and scale<br>"
    "<b>affine</b> - Translation, rotation, scale, and shear<br>"
    "<b>nl</b> - Non-linear registration<br>"
    "<br>The suffix of the registration also indicates the type:<br>"
    "<b>reduced</b> - Quicker registration with fewer parameters or optimization steps<br>"
    "<b>mid</b> - Medium registration with more parameters or optimization steps<br>"
    "<b>expanded</b> - Slower registration with more optimization steps<br>"
    "<b>extreme</b> - Even slower registration with more optimization steps<br>"
    "<br>Additionally, these suffixes will indicate which metric is used during the registration<br>"
    "<b>ams</b> - Advanced Mean Squares metric instead of Advanced Mattes Mutual Information<br>"
    "<b>anc</b> - Advanced Normalized Correlation metric instead of Advanced Mattes Mutual Information<br>"
)

# to add apps: volume viewer, sync viewer,
REGISTER_TEXT = "Co-register your microscopy and imaging mass spectrometry data."
VIEWER_TEXT = "Overlay your microscopy and imaging mass spectrometry data."
CROP_TEXT = "Crop your microscopy data to reduce it's size (handy for Image Fusion)."
CONVERT_TEXT = "Convert multi-scene CZI images or other formats to OME-TIFF."
MERGE_TEXT = "Merge multiple OME-TIFF images into a single file."
FUSION_TEXT = "Export your data for Image Fusion in MATLAB compatible format."
ELASTIX_TEXT = "Register whole slide microscopy images<br>(<b>Elastix</b>)."
VALIS_TEXT = "Register whole slide microscopy images<br>(<b>Valis</b>)."
RUNNER_TEXT = "Queue and run whole slide registration projects<br>(<b>Elastix/Valis</b>)."
CONVERT_WARNING = ""
if not STATE.allow_convert:
    CONVERT_WARNING = "<i>Not available on Apple Silicon due to a bug I can't find...</i>"
VALIS_WARNING = ""
if not STATE.allow_valis:
    VALIS_WARNING = "<br><br><i>Might not work without a proper setup.</i>"

EXPERIMENTAL_WARNING = "This plugin is experimental and may not work as intended. Please report any issues."


MASKS_HELP = (
    "Optional: compare matching tissue regions in two segmentation masks using Dice and IoU."
    "<br>Load one mask for each image, then pair the labels representing the same region. Background 0 is excluded."
    "<br>Masks are sampled on the fixed grid; this does not create segmentations."
)
LANDMARK_HELP = (
    "<p><b>Independent landmark accuracy</b></p>\n"
    "<p>Measures the distance between corresponding landmarks after applying the registration.</p>\n"
    "<p><b>Units and frame:</b> um in the moving image.<br>"
    "<b>Residual:</b> G(fixed) - moving.<br>"
    "<b>Evaluated / eligible:</b> usable measurements out of the eligible landmarks.</p>\n"
    "<p>Mean, median, RMS, P95, and maximum summarize the measured distances. "
    "Known fitting landmarks are excluded.</p>\n"
    "<p>No universal pass/fail threshold is applied. Accuracy between landmarks is not inferred.</p>"
)
FITTING_HELP = (
    "<p><b>Original fitting residuals</b></p>\n"
    "<p>These points were used to estimate the registration. Their residuals describe the fit to those points; "
    "they are <b>not independent validation accuracy</b>.</p>\n"
    "<p><b>Units:</b> um in the moving image.<br>"
    "<b>Availability:</b> requires the original fitting landmarks.</p>\n"
    "<p>Register's original <b>root-sum-square</b> error is preserved. It increases with point count.</p>"
)
OVERLAP_HELP = (
    "<p><b>Corresponding region overlap</b></p>\n"
    "<p>Compares matching regions in two segmentation masks after registration.</p>\n"
    "<p><b>Dice:</b> 2 x intersection / summed areas.<br>"
    "<b>IoU:</b> intersection / union.<br>"
    "Both range from 0 to 1; higher means more overlap.</p>\n"
    "<p><b>Areas:</b> um2 on the fixed analysis grid.<br>"
    "<b>Coverage:</b> the available mask fields of view.</p>\n"
    "<p>Background label 0 is excluded. Both-empty labels are unavailable; one-empty labels score 0. "
    "Use a finer analysis grid for small regions.</p>"
)
DOWNSAMPLE_HELP = (
    "<p><b>Analysis resolution</b></p>\n"
    "<p>Applies to the <b>selected image pair</b>. Larger downsample values use a coarser grid.</p>\n"
    "<p>The fixed analysis grid is capped by <b>Max grid size</b> (default 4 megapixels, up to 64), so its "
    "actual spacing may be coarser than requested. Finer grids improve small-region and deformation sampling, "
    "but need roughly 70 bytes per grid pixel per registration candidate during analysis.</p>\n"
    "<p><b>Landmarks:</b> always measured at full physical precision.<br>"
    "<b>Image display:</b> native pyramids and zoom are independent of this setting.</p>"
)
DEFORMATION_HELP = (
    "<p><b>Transform geometry</b></p>\n"
    "<p>Describes local area change and orientation in the imported transformation. "
    "It does <b>not measure registration accuracy</b>.</p>\n"
    "<p><b>Source coverage:</b> fixed-grid positions mapping inside the moving image.<br>"
    "<b>Derivative coverage:</b> positions with a usable local derivative.<br>"
    "<b>Jacobian:</b> fixed-to-moving area scaling.<br>"
    "<b>Area ratio:</b> moving-to-fixed area scaling.</p>\n"
    "<p>Global affine scaling is included. Nonpositive Jacobians may include intentional reflections.</p>\n"
    "<p><b>Images without an imported transform:</b> the identity mapping is used; "
    "historical registration deformation is unknown.</p>"
)
LOCALIZATION_HELP = (
    "<p><b>Localization (click) error</b></p>\n"
    "<p>A measured landmark error is registration error <b>plus</b> the error of locating the landmark in each "
    "image. Adding random noise to clicks cannot separate the two, so localization error is measured directly:</p>"
    "\n<p><b>Re-click landmarks:</b> a random subset of manual landmarks is shown again with markers hidden, inside "
    "a jittered search circle. The spread of repeat clicks measures your precision (not systematic bias).<br>"
    "<b>Calibrate clicking (decoys):</b> a marked feature must be found in a copy of the same image shifted by a "
    "hidden amount. The error against the known answer includes bias, but has no cross-modal ambiguity, so it is a "
    "lower bound.<br>"
    "<b>Matcher jitter</b> (Analysis settings): assisted and automatic landmarks are re-matched with jittered "
    "position, template size and search start, using the automatic template (48 px) and search (24 px) sizes.</p>\n"
    "<p>Re-clicks are used when available, otherwise decoys. If only one image was measured, the other is assumed "
    "to have the same precision in pixels. Manual landmarks use the click precision in both images, assisted "
    "landmarks combine fixed click and matcher precision, and automatic landmarks use matcher precision.</p>\n"
    "<p><b>Corrected RMS:</b> sqrt(observed RMS^2 - localization RMS^2), the registration error left after "
    "removing localization error. <b>Within noise:</b> the residual is within the 95% range expected from "
    "localization alone, so it shows no detectable registration error.</p>\n"
    "<p>Re-clicks and decoys are saved in the study file but not in landmark CSV exports.</p>"
)
DIRECTION_HELP = (
    "<p><b>Directional bias</b></p>\n"
    "<p>Shows whether residuals prefer the horizontal (X) or vertical (Y) direction, both, or neither.</p>\n"
    "<p><b>Left:</b> each residual as a point (dX, dY), with Y pointing down as in the image. "
    "Dashed circle: RMS error. Red ellipse: 95% covariance. Red arrow: mean bias. "
    "Blue region: 95% localization (click) noise, when measured.<br>"
    "<b>Right:</b> residual directions weighted by magnitude; 0 deg is +X and 90 deg points down.</p>\n"
    "<p><b>Verdict:</b> with measured localization noise, an axis has error when its residuals exceed that noise "
    "(chi-square, 5%). Without it, an RMS ratio of 1.5 decides horizontal or vertical, and 'none' cannot be "
    "concluded. <b>Oblique:</b> the error is elongated along a diagonal.</p>\n"
    "<p>Landmark residuals are in the moving frame; shift tiles are in the fixed frame. Overlapping tiles are not "
    "independent, so their p-values are optimistic.</p>"
)
DIAGNOSTIC_HELP = {
    "overlap": (
        "<p><b>Image overlap</b></p>\n"
        "<p>Visual comparison on the fixed analysis grid.</p>\n"
        "<p><b>Magenta:</b> fixed image.<br>"
        "<b>Green:</b> registered moving image.<br>"
        "<b>Gray:</b> intensity agreement.</p>\n"
        "<p>Uses the saved scalar analysis channels, independently of native RGB display settings. "
        "Intensity agreement alone does not establish registration accuracy.</p>"
    ),
    "validation_errors": (
        "<p><b>Independent landmark errors</b></p>\n"
        "<p><b>Point color:</b> measured positional error in um; see the colorbar.<br>"
        "<b>Cyan arrows:</b> observed moving point to the predicted position, at actual physical length.<br>"
        "<b>Background:</b> the native moving image for context.</p>\n"
        "<p><b>Residual:</b> G(fixed) - moving, in the moving physical frame. "
        "Known fitting landmarks are excluded.</p>\n"
        "<p>Accuracy between landmarks is not inferred. No universal pass/fail threshold is applied.</p>"
    ),
    "mask_disagreement": (
        "<p><b>Mask disagreement</b></p>\n"
        "<p>Requires two segmentation masks and corresponding positive labels. Background 0 is excluded.</p>\n"
        "<p><b>Green:</b> agreement.<br>"
        "<b>Magenta:</b> fixed region only.<br>"
        "<b>Cyan:</b> moving region only.</p>\n"
        "<p>Missing masks skip region measurements; other analyses can still run.</p>"
    ),
    "area_change": (
        "<p><b>Local area change</b></p>\n"
        "<p>Shows log2 of the <b>moving-to-fixed</b> area ratio.</p>\n"
        "<p><b>0:</b> unchanged area.<br>"
        "<b>Positive:</b> expansion (+1 doubles area).<br>"
        "<b>Negative:</b> compression (-1 halves area).</p>\n"
        "<p><b>Gray:</b> unavailable.<br>"
        "<b>Purple:</b> nonpositive Jacobian.</p>\n"
        "<p>The area ratio is the reciprocal of a positive fixed-to-moving Jacobian determinant. "
        "Global affine scaling is included.</p>"
    ),
    "signed_jacobian": (
        "<p><b>Signed Jacobian determinant</b></p>\n"
        "<p>Local area scaling of the complete <b>fixed-to-moving</b> map in physical coordinates.</p>\n"
        "<p><b>1:</b> unchanged local area.<br>"
        "<b>Above 1:</b> expansion in this direction.<br>"
        "<b>Between 0 and 1:</b> compression in this direction.<br>"
        "<b>0 or negative:</b> singularity or orientation reversal.<br>"
        "<b>Gray:</b> no usable derivative or source coverage.</p>\n"
        "<p>Includes global affine scaling. Intentional reflections can produce negative values.</p>"
    ),
    "nonpositive_jacobian": (
        "<p><b>Nonpositive Jacobian</b></p>\n"
        "<p>Marks locations with a Jacobian determinant of <b>0 or below</b>.</p>\n"
        "<p><b>Fraction:</b> affected locations divided by locations with usable derivatives and source coverage.<br>"
        "<b>Gray:</b> unavailable locations.</p>\n"
        "<p>Intentional reflections can contribute. This is not automatically a tissue-folding "
        "or registration-failure score.</p>"
    ),
    "residual_direction": (
        "<p><b>Residual direction map</b></p>\n"
        "<p>Colors each reliable residual shift tile by the direction of its misalignment (hue, see the wheel) and "
        "its size (brightness, full at the 95th percentile). Black means no measurable shift; gray tiles were "
        "rejected.</p>\n"
        "<p>Neighbouring tiles with the same color indicate a coherent local shift; changing colors indicate "
        "rotation or local deformation. Fixed frame. Requires Residual shift map under Automatic evidence.</p>"
    ),
    "residual_shift": (
        "<p><b>Residual shift map</b></p>\n"
        "<p>Local misalignment between the fixed and registered moving images, estimated by phase correlation "
        "on overlapping tiles of the analysis grid.</p>\n"
        "<p><b>Color:</b> shift magnitude in um, <b>fixed frame</b>.<br>"
        "<b>Cyan arrows:</b> predicted minus observed position for each reliable tile.<br>"
        "<b>Gray:</b> tiles rejected for low texture, incomplete coverage, or low correlation.</p>\n"
        "<p>Image-similarity evidence, not independent ground truth. Cross-modal or low-texture images "
        "are reported as unavailable rather than accurate.</p>"
    ),
    "automatic_landmarks": (
        "<p><b>Automatic landmarks</b></p>\n"
        "<p>Landmarks added with <b>Detect landmarks</b> in the Landmarks tab: SIFT correspondences between "
        "the fixed and registered images, refined by template matching at native resolution.</p>\n"
        "<p><b>Point color:</b> residual G(fixed) - moving in um, <b>moving frame</b>, as for manual landmarks.<br>"
        "<b>Background:</b> the native moving image.</p>\n"
        "<p>Reported separately from manual and assisted landmarks and never pooled with them. Unticked "
        "landmarks, including detected outliers, are not shown.</p>"
    ),
}
AUTOMATIC_HELP = (
    "<p><b>Automatic evidence</b></p>\n"
    "<p>Measurements computed from image content, reported separately from manual and assisted landmarks. "
    "They compare the fixed image with the moving image resampled through each registration, so only "
    "remaining local misalignment is measured.</p>\n"
    "<p><b>Residual shift map:</b> optional per-tile shift in um (fixed frame), with valid-tile coverage. "
    "<b>Tile</b> is the correlation window; Auto uses 64 grid pixels.\n"
    "<b>Automatic landmarks:</b> added and reviewed in the Landmarks tab with Detect landmarks; per-point "
    "residuals in um (moving frame).</p>\n"
    "<p>These measurements depend on image similarity: they can fail, or be biased, for different stains or "
    "modalities, repetitive texture, or tissue damage. They complement, but do not replace, independent "
    "landmarks.</p>"
)
ASSIST_HELP = (
    "<p><b>Assisted placement</b></p>\n"
    "<p>After you click the fixed image, the moving image is resampled through the selected registration "
    "and searched for the same content within the search radius.</p>\n"
    "<p><b>Enter</b> accepts the suggestion; the landmark is stored as <b>assisted</b> with its correlation "
    "score. Clicking the moving image places the point yourself and stores it as <b>manual</b>.</p>\n"
    "<p><b>Template:</b> Auto uses 48 fixed pixels. <b>Search radius:</b> Auto uses 24 fixed pixels. "
    "Use a radius larger than the errors you expect; a suggestion at the edge of the search is rejected.</p>\n"
    "<p><b>Detect landmarks</b> finds up to <b>Max</b> matching features spread across the image and adds them "
    "as <b>automatic</b> landmarks (cyan) for review; outliers are added unticked. They are reported separately "
    "from manual and assisted landmarks. Ctrl+Z removes a whole detection; Remove automatic deletes them all.</p>"
)
LANDMARK_TABLE_HELP = (
    "<p><b>Landmark table and views</b></p>\n"
    "Table and views use um. Add landmarks by clicking fixed, then moving; adding continues until Esc. "
    "Drag existing points or edit coordinates below."
    "<p>\nShortcuts:\n</p>"
    "<b>A</b> add\n"
    "<b>Esc</b> stop\n"
    "<b>Del</b> delete\n"
    "<b>Ctrl+Z</b> undo."
)

LINKS = {
    "micro-sam": "https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#choosing-a-model",
}
