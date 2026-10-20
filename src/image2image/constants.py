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

LANDMARK_HELP = (
    "Residual = G(fixed) - moving, measured in um in the moving physical frame."
    "<br>Evaluated/eligible counts  show how many independent landmarks were usable. "
    "<br>Mean, median, RMS, P95, and maximum describe  those measured locations. Known fitting landmarks are excluded."
    "<br> No universal pass/fail threshold is applied."
)
FITTING_HELP = (
    "Fitting residuals describe the original points used to estimate a registration, when available."
    "<br>They measure fit to training evidence and are separate from independent validation accuracy."
    "<br>Register's original root-sum-square error is preserved and increases with point count.",
)
OVERLAP_HELP = (
    "Dice = 2 * intersection / summed areas; IoU = intersection / union. Both range from zero to one."
    "<br>Areas use um2 on the fixed analysis grid. Coverage describes available mask fields of view."
    "<br>Background zero is excluded, both-empty labels are unavailable, and one-empty labels score zero."
    "<br>Small regions may require a finer analysis grid.",
)
DOWNSAMPLE_HELP = (
    "This setting applies to the selected moving-image pair. The common fixed analysis grid is "
    "<br>capped at 4 million pixels, so its actual spacing may be coarser. Finer grids improve "
    "<br>small-region  and deformation sampling. Landmark residuals always use full-precision physical"
    "<br>coordinates; native viewer pyramids and zoom are independent of this setting."
)
DIAGNOSTIC_HELP = {
    "overlap": (
        "Image overlap is a visual comparison on the bounded fixed analysis grid: fixed is magenta, registered moving "
        "is green, and agreement is gray."
        "<br>It uses the saved scalar analysis channels, not native RGB display settings. "
        "<br>Intensity agreement alone does not establish registration accuracy."
    ),
    "validation_errors": (
        "Validation errors are independent landmark residuals G(fixed) - moving, shown in the moving physical frame "
        "in um."
        "<br>Points show measured distances; arrows show residual direction and magnitude. Known fitting points "
        "are excluded."
        "<br>No error is inferred between landmarks and no universal pass/fail threshold is applied."
    ),
    "mask_disagreement": (
        "Mask disagreement compares corresponding positive segmentation labels after registration."
        "<br>Green means  agreement, magenta fixed only, and cyan moving only."
        "<br>Background label zero is excluded. Both masks and  label correspondences are required; missing masks do "
        "not prevent other measurements."
    ),
    "area_change": (
        "Area change is log2 of the moving-to-fixed area ratio, the reciprocal of a positive fixed-to-moving "
        "Jacobian determinant. "
        "<br>Zero means unchanged area, positive expansion, and negative compression. "
        "<br>Maps include global affine scaling. Gray is unavailable and purple is nonpositive Jacobian."
    ),
    "signed_jacobian": (
        "Signed Jacobian is the determinant of the derivative of the complete fixed-to-moving map in physical "
        "coordinates. "
        "<br>One means unchanged local area. Nonpositive values indicate an orientation reversal or "
        "singularity and may include intentional reflections. "
        "<br>Gray regions have no usable derivative or source coverage."
    ),
    "nonpositive_jacobian": (
        "Nonpositive Jacobian marks sampled locations with determinant at or below zero. "
        "<br>The reported fraction uses locations with valid derivatives and source coverage. "
        "<br>Intentional reflections can contribute;  this is not automatically a tissue-folding or"
        " registration-failure score."
    ),
}
