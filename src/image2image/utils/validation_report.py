"""Self-contained visual reports and machine-readable validation exports."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import re
import shutil
import textwrap
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from image2image.models.validation import CandidateAnalysis, StudyAnalysis, ValidationAnalysis, ValidationSession
from image2image.utils.validation import CHI2_2D_95, direction_vectors

_METHODS = (
    (
        "Independent landmark accuracy",
        "Residuals are G(fixed) - moving in moving-image physical coordinates (um). "
        "Error is the Euclidean length of each residual. Mean, median, RMS, 95th percentile (P95), "
        "and maximum summarize finite, included errors; RMS is the square root of the mean squared error. "
        "Disabled points, points outside either native image, and known fitting landmarks from any candidate "
        "are excluded from every candidate's validation set. Fitting-point reuse is checked in either image "
        "with a tolerance of 0.000001 um. Non-finite transformed points are also excluded. "
        "Used and eligible counts are explicit. Other validation independence is user-declared.",
    ),
    (
        "Fitting evidence and interpretation",
        "Fitting residuals summarize the original registration landmarks separately from independent validation. "
        "Unavailable evidence is not represented as zero error. Sparse landmark measurements do not establish "
        "accuracy everywhere. No universal pass/fail threshold is applied. Measurements are not pooled across image pairs.",
    ),
    (
        "Sampling and calibration",
        "All candidates for a pair use one fixed-image grid. The integer sampling factor is at least the requested "
        "downsample and increases as needed to respect the pixel cap. Spacing and origin are calibrated in um. "
        "Images use linear interpolation; label masks use nearest-neighbor sampling. "
        "Image fingerprints use file size and modification time; transform definitions are fingerprinted by content.",
    ),
    (
        "Region overlap",
        "Corresponding non-background labels are compared on the sampled fixed grid. "
        "Dice = 2 * intersection / (fixed area + moving area); IoU = intersection / union. "
        "Areas are pixel counts multiplied by grid pixel area (um2). Both-empty masks are unavailable; "
        "one-empty masks score zero. Moving content outside the fixed field is clipped. "
        "Source and joint coverage are reported. Coarse sampling can miss small regions.",
    ),
    (
        "Deformation and area change",
        "The signed Jacobian determinant describes the complete fixed-to-moving map, including global affine scaling. "
        "Derivatives use central differences in physical coordinates. Outer grid boundaries, unsupported source samples, "
        "and non-finite derivatives are excluded. For positive determinants, moving-to-fixed area ratio = 1 / determinant "
        "and log2 area ratio = -log2(determinant). Positive values indicate expansion; negative values indicate compression. "
        "The nonpositive fraction uses valid derivative samples as its denominator. Nonpositive determinants can indicate "
        "orientation reversal or singularity, including intended reflections. Deformation is not measured accuracy. "
        "Already aligned images use an identity mapping in calibrated physical space; original registration deformation is unknown.",
    ),
    (
        "Automatic evidence",
        "Optional measurements derived from image content, reported separately from independent landmarks and "
        "never pooled with them. Both compare the fixed image with the moving image resampled through each "
        "registration on the fixed analysis grid. The residual shift map estimates per-tile shifts by phase "
        "correlation with sub-pixel refinement; tiles with incomplete coverage, low texture, post-shift correlation "
        "below 0.5, or shifts above a quarter tile are rejected, and shifts are reported in the fixed frame. "
        "Automatic landmarks are detected in the Landmarks tab and reviewed by the user: SIFT correspondences "
        "between the fixed image and the moving image registered through the selected registration, refined by "
        "normalized cross-correlation at native resolution. Outliers above the median plus 3.5 robust standard "
        "deviations are added unticked. Their residuals use the landmark definition G(fixed) - moving in the "
        "moving frame and are summarized separately; headline accuracy uses manual and assisted landmarks only, "
        "which are the independent evidence. Because detection searches near the selected registration's "
        "prediction, comparing registrations on automatic landmarks can favour the one used for detection. "
        "Both depend on image similarity and can fail or be biased for different stains or "
        "modalities, repetitive texture, or tissue damage; too few reliable measurements are reported as "
        "unavailable, not as zero error.",
    ),
    (
        "Localization error",
        "Observed landmark residuals combine registration error with the error of locating each landmark "
        "(localization or click error). Localization precision is measured by blind repeat clicks (pooled "
        "within-landmark standard deviation; precision only), by decoy trials in which a feature is found in a copy "
        "of the same image shifted by a hidden known amount (RMS error including bias; a lower bound because there is "
        "no cross-modal ambiguity), and for assisted and automatic landmarks by re-running the template matcher with "
        "random jitter of position, template size and search start. Fixed-image precision is propagated through the "
        "local Jacobian J of the registration: expected localization covariance = J S_fixed J^T + S_moving. "
        "Corrected RMS = sqrt(max(0, observed RMS^2 - mean localization variance)) estimates the registration error "
        "alone. A residual is within localization noise when its squared Mahalanobis distance is below the 95% "
        "chi-square quantile with 2 degrees of freedom.",
    ),
    (
        "Directional bias",
        "Residual vectors are summarized separately for independent landmarks and automatic landmarks "
        "(G(fixed) - moving, moving frame) and for reliable residual shift tiles (fixed frame); frames are never mixed. "
        "Mean bias is tested against zero with Hotelling's T-squared test. When localization noise has been measured, "
        "each axis is reported as having error when its sum of squared residuals exceeds that noise (chi-square, 5% "
        "level), giving none, horizontal, vertical or both. Otherwise the verdict uses an RMS ratio of 1.5 and cannot "
        "conclude 'none'. Oblique marks an elongated covariance with its principal axis 30 to 60 degrees from an "
        "image axis. Overlapping shift tiles are not independent, so their p-values are optimistic. "
        "Residual direction maps color shift tiles by direction (hue) and magnitude (brightness).",
    ),
    (
        "Diagnostic figures",
        "Fixed images appear magenta and registered moving images green. Landmark colors show measured error; arrows "
        "show residuals at their physical length in the moving frame. Area-change colors use a symmetric log2 scale; "
        "purple marks nonpositive determinants and gray marks unavailable samples. Mask agreement is green, fixed-only "
        "magenta, and moving-only cyan. Error and area color limits are shared across candidates within each pair. "
        "Image previews use 1st to 99th percentile contrast and may be reduced for display; numerical measurements are unchanged.",
    ),
)

_LABELS = {
    "validation_count": "Used landmarks",
    "validation_eligible": "Eligible landmarks",
    "validation_mean_um": "Mean (um)",
    "validation_median_um": "Median (um)",
    "validation_rms_um": "RMS (um)",
    "validation_p95_um": "P95 (um)",
    "validation_maximum_um": "Maximum (um)",
    "rms": "RMS",
    "p95": "P95",
    "iou": "IoU",
    "coverage": "Source coverage",
    "nonpositive_fraction": "Nonpositive Jacobian fraction (valid samples)",
    "transform_reference": "Transform reference",
}


def _label(key: str) -> str:
    return _LABELS.get(key, key.replace("_um2", " (um2)").replace("_um", " (um)").replace("_", " ").capitalize())


def _format(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _redact_paths(value: Any, sources: tuple[str, ...] = ()) -> Any:
    """Remove directory components from structured paths and paths embedded in messages."""
    if isinstance(value, dict):
        return {_redact_paths(key, sources): _redact_paths(item, sources) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_redact_paths(item, sources) for item in value]
    if not isinstance(value, str):
        return value
    for source in sources:
        value = value.replace(source, source.replace("\\", "/").rsplit("/", 1)[-1])
    # Whole paths and quoted paths can contain spaces. Also handle Windows paths on POSIX hosts.
    if value.startswith(("/", "\\", "~/", "./", "../")) or re.match(r"^[A-Za-z]:[\\/]", value):
        return value.replace("\\", "/").rsplit("/", 1)[-1]
    value = re.sub(
        r"([\"'])([^\"'\n]*[/\\][^\"'\n]*)\1",
        lambda match: match[1] + match[2].replace("\\", "/").rsplit("/", 1)[-1] + match[1],
        value,
    )
    return re.sub(
        r"(?<![\w/\\])(?:[A-Za-z]:[/\\]|[/\\]|\.{1,2}[/\\]|~[/\\])[^\s<>\"'|]+"
        r"|(?<![\w/\\])(?:[\w.-]+[/\\])+[\w.-]+\.[a-zA-Z0-9]+",
        lambda match: match[0].replace("\\", "/").rsplit("/", 1)[-1],
        value,
    )


def _source_paths(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        paths = [item for key, item in value.items() if key in ("path", "source") and isinstance(item, str) and item]
        paths.extend(path for item in value.values() for path in _source_paths(item))
        # Also omit known directories when a failure refers to a different file in the same directory.
        for path in tuple(paths):
            separator = "\\" if "\\" in path else "/"
            if separator in path:
                directory = path.rsplit(separator, 1)[0] + separator
                if directory.strip("/\\"):
                    paths.append(directory)
        return tuple(sorted(set(paths), key=len, reverse=True))
    if isinstance(value, list | tuple):
        return tuple(path for item in value for path in _source_paths(item))
    if isinstance(value, str) and (value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[/\\]", value)):
        return (value,)
    return ()


def _report_snapshot(analysis: ValidationAnalysis, sources: tuple[str, ...]) -> ValidationAnalysis:
    session = ValidationSession.model_validate(_redact_paths(analysis.session.model_dump(mode="json"), sources))
    candidates = []
    for candidate in analysis.candidates:
        record = asdict(candidate)
        record.pop("map_path")
        candidates.append(CandidateAnalysis(**_redact_paths(record, sources), map_path=candidate.map_path))
    return replace(analysis, session=session, candidates=candidates, versions=_redact_paths(analysis.versions, sources))


def comparison_rows(analysis: ValidationAnalysis) -> list[dict[str, Any]]:
    """Return comparison rows with explicit metric names, units, and denominators."""
    return [
        {
            "candidate": result.name,
            "status": result.status,
            "transform_reference": result.deformation.get("reference", "Imported registration transform."),
            "validation_count": result.validation.get("count", 0),
            "validation_eligible": result.validation.get("eligible", 0),
            **{
                f"validation_{key}_um": result.validation.get(key)
                for key in ("mean", "median", "rms", "p95", "maximum")
            },
            "fitting_rms_um": result.fitting.get("rms"),
            "coverage": result.deformation.get("coverage"),
            "nonpositive_fraction": result.deformation.get("nonpositive_fraction"),
            "automatic_count": result.automatic.get("count"),
            "automatic_eligible": result.automatic.get("eligible"),
            "automatic_median_um": result.automatic.get("median"),
            "residual_map_valid_fraction": result.residual_map.get("valid_fraction"),
            "residual_map_median_um": result.residual_map.get("median"),
            "localization_rms_um": result.localization.get("independent", {}).get("fle_rms"),
            "corrected_rms_um": result.localization.get("independent", {}).get("corrected_rms"),
        }
        for result in analysis.candidates
    ]


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>Unavailable: no evidence supplied.</p>"
    keys = list(rows[0])
    headings = "".join(f"<th scope='col'>{html.escape(_label(key))}</th>" for key in keys)
    body = []
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key)
            text = _format(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table'><table><thead><tr>{headings}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or (list(rows[0]) if rows else ["candidate"]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(values: dict[str, Any]) -> str:
    return _table([{"metric": _label(key), "value": value} for key, value in values.items()])


def _comparison_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "candidate",
        "status",
        "validation_count",
        "validation_eligible",
        "validation_mean_um",
        "validation_median_um",
        "validation_rms_um",
        "validation_p95_um",
        "validation_maximum_um",
    ]
    if any(row.get("automatic_count") is not None for row in rows):
        columns += ["automatic_count", "automatic_median_um"]
    if any(row.get("residual_map_valid_fraction") is not None for row in rows):
        columns += ["residual_map_valid_fraction", "residual_map_median_um"]
    if any(row.get("corrected_rms_um") is not None for row in rows):
        columns += ["localization_rms_um", "corrected_rms_um"]
    return _table([{key: row.get(key) for key in columns} for row in rows])


def _input_rows(analysis: ValidationAnalysis) -> list[dict[str, Any]]:
    rows = []
    for role in ("fixed", "moving", "fixed_mask", "moving_mask"):
        image = getattr(analysis.session, role)
        if image is not None:
            rows.append(
                {
                    "input": _label(role),
                    "file": image.path,
                    "shape_yx": image.geometry.shape,
                    "spacing_xy_um": image.geometry.spacing,
                    "origin_xy_um": image.geometry.origin,
                    "channel": image.channel,
                    "calibration_source": image.calibration_source,
                    "calibration_confirmed": image.calibration_confirmed,
                }
            )
    return rows


def _settings(analysis: ValidationAnalysis) -> dict[str, Any]:
    return {
        "created": analysis.created,
        "grid_shape_yx": analysis.grid.shape,
        "grid_spacing_xy_um": analysis.grid.spacing,
        "grid_origin_xy_um": analysis.grid.origin,
        "requested_downsample": analysis.session.downsample,
        "maximum_grid_pixels": analysis.session.max_pixels,
        "validation_independence": analysis.session.independence_note,
    }


def _context(analysis: ValidationAnalysis) -> str:
    candidates = [candidate.model_dump() for candidate in analysis.session.candidates]
    return (
        "<details><summary>Inputs, calibration and analysis settings</summary>"
        f"<h3>Image and mask inputs</h3>{_table(_input_rows(analysis))}"
        f"<h3>Analysis grid and settings</h3>{_metrics(_settings(analysis))}"
        f"<h3>Registration candidates</h3>{_table(candidates)}</details>"
    )


def _candidate_section(
    candidate: CandidateAnalysis, png: bytes, level: int = 2, direction_png: bytes | None = None
) -> str:
    heading = level + 1
    return (
        f"<section><h{level}>{html.escape(candidate.name)}</h{level}>"
        f"<p><strong>Status:</strong> {html.escape(candidate.status)}</p>"
        + (f"<p class='notice'>{html.escape(candidate.error)}</p>" if candidate.error else "")
        + f"<h{heading}>Independent landmark accuracy (um)</h{heading}>{_metrics(candidate.validation)}"
        "<figure><img alt='Image overlap, landmark residuals, area change and mask disagreement' "
        f"src='data:image/png;base64,{base64.b64encode(png).decode()}'>"
        "<figcaption>Diagnostic views. See Methods and interpretation for color keys and measurement limits.</figcaption></figure>"
        f"<h{heading}>Fitting evidence (um, moving frame)</h{heading}>{_metrics(candidate.fitting)}"
        f"<h{heading}>Region overlap on the fixed grid</h{heading}>{_table(candidate.masks)}"
        f"<p class='notice'>{html.escape(candidate.mask_reason)}</p>"
        f"<h{heading}>Deformation</h{heading}>{_metrics(candidate.deformation)}"
        + _localization_section(candidate, heading)
        + _direction_section(candidate, direction_png, heading)
        + _automatic_section(candidate, heading)
        + "<details><summary>Per-landmark residuals and exclusions</summary>"
        f"{_table(candidate.landmarks)}</details></section>"
    )


def _localization_section(candidate: CandidateAnalysis, heading: int) -> str:
    rows = localization_rows(candidate.localization)
    if not rows:
        return ""
    return f"<h{heading}>Localization error and corrected accuracy (um, moving frame)</h{heading}>{_table(rows)}"


def _direction_section(candidate: CandidateAnalysis, png: bytes | None, heading: int) -> str:
    if not candidate.direction:
        return ""
    text = f"<h{heading}>Directional bias</h{heading}>"
    for key, label in DIRECTION_SOURCES.items():
        stats = candidate.direction.get(key)
        if stats and stats.get("count"):
            text += f"<p><strong>{html.escape(label)}:</strong> {html.escape(direction_text(stats))}</p>"
    if png is not None:
        text += (
            "<figure><img alt='Residual target plots and direction rose diagrams' "
            f"src='data:image/png;base64,{base64.b64encode(png).decode()}'>"
            "<figcaption>Left: residual vectors with RMS circle, 95% covariance ellipse (red), mean bias arrow and, "
            "when measured, 95% localization noise (blue). Y points down as in the image. Right: residual "
            "directions weighted by magnitude; 90 deg points down.</figcaption></figure>"
        )
    return text


def _automatic_section(candidate: CandidateAnalysis, heading: int) -> str:
    if not candidate.residual_map and not candidate.automatic:
        return ""
    text = f"<h{heading}>Automatic evidence (image similarity; not independent ground truth)</h{heading}>"
    if candidate.residual_map:
        text += (
            f"<h{heading + 1}>Residual shift map (um, fixed frame)</h{heading + 1}>{_metrics(candidate.residual_map)}"
        )
    if candidate.automatic:
        text += (
            f"<h{heading + 1}>Automatic landmarks (um, moving frame)</h{heading + 1}>{_metrics(candidate.automatic)}"
        )
        rows = [row for row in candidate.landmarks if row.get("source") == "automatic"]
        if rows:
            text += f"<details><summary>Automatic landmark residuals</summary>{_table(rows)}</details>"
    return text


def _document(title: str, body: str, include_paths: bool) -> str:
    methods = "\n".join(f"<h3>{title}</h3>\n<p>{text}</p>" for title, text in _METHODS)
    privacy = (
        "Source directory paths included." if include_paths else "Source directory paths omitted; filenames retained."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font:15px/1.6 system-ui,sans-serif;max-width:1240px;margin:40px auto;padding:0 24px;color:#203040;background:#fff}}
h1,h2,h3,h4{{line-height:1.25;color:#163f55}}h1{{font-size:30px}}h2{{margin-top:32px}}
header{{border-bottom:3px solid #267089;padding-bottom:16px}}section{{border-top:1px solid #d2dce2;margin-top:28px;padding-top:8px}}
p,li{{white-space:pre-line}}.notice,figcaption{{color:#526575}}.notice:empty{{display:none}}
.table{{overflow-x:auto;margin:16px 0}}table{{border-collapse:collapse;font-size:13px;width:100%}}
td,th{{padding:9px 12px;border:1px solid #dce4e9;text-align:left;vertical-align:top;white-space:pre-line;overflow-wrap:anywhere}}
th{{background:#edf3f6;font-weight:650}}tbody tr:nth-child(even){{background:#f8fafb}}
details{{margin:22px 0;border:1px solid #d2dce2;border-radius:6px;padding:12px 16px}}
summary{{cursor:pointer;font-weight:650;color:#163f55}}pre{{overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}
figure{{margin:20px 0}}img{{width:100%;height:auto}}figcaption{{font-size:13px}}a{{color:#176681}}
@media print{{body{{margin:0;padding:0}}details{{display:block}}.table{{overflow:visible}}}}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1>
<p><strong>Independent accuracy, fitting evidence, region overlap and deformation.</strong><br>
Each measurement describes its own evidence; no universal pass/fail threshold is applied.</p>
<p class="notice">{privacy} <a href="report.pdf">Download the PDF report</a></p></header>
<details><summary>Methods and interpretation</summary>
{methods}
</details>
{body}
</body>
</html>
"""


def _pdf_pair_sections(analysis: ValidationAnalysis) -> list[tuple[str, str]]:
    sections = [
        (
            "Analysis settings",
            "\n".join(f"{_label(key)}: {_format(value)}" for key, value in _settings(analysis).items()),
        )
    ]
    for row in _input_rows(analysis):
        sections.append(
            (
                f"Input: {row['input']}",
                "\n".join(f"{_label(key)}: {_format(value)}" for key, value in row.items() if key != "input"),
            )
        )
    for definition in analysis.session.candidates:
        sections.append(
            (
                f"Registration: {definition.name}",
                "\n".join(f"{_label(key)}: {_format(value)}" for key, value in definition.model_dump().items()),
            )
        )
    candidate_sections = []
    for candidate in analysis.candidates:
        lines = [f"Status: {candidate.status}"]
        if candidate.error:
            lines.append(candidate.error)
        for name, values in (
            ("Independent validation (um)", candidate.validation),
            ("Fitting evidence (um)", candidate.fitting),
            ("Deformation", candidate.deformation),
            ("Residual shift map (um, fixed frame)", candidate.residual_map),
            ("Automatic landmarks (um, moving frame)", candidate.automatic),
        ):
            if not values:
                continue
            lines.append(name + ": " + "; ".join(f"{_label(key)}: {_format(value)}" for key, value in values.items()))
        for row in localization_rows(candidate.localization):
            lines.append(
                f"Localization, {row['statistic']}: independent {_format(row['independent'])}; "
                f"automatic {_format(row['automatic'])}"
            )
        for key, label in DIRECTION_SOURCES.items():
            stats = candidate.direction.get(key)
            if stats and stats.get("count"):
                lines.append(f"Directional bias, {label}: {direction_text(stats)}")
        if candidate.mask_reason:
            lines.append("Region overlap: " + candidate.mask_reason)
        for mask in candidate.masks:
            lines.append(
                "Region overlap: " + "; ".join(f"{_label(key)}: {_format(value)}" for key, value in mask.items())
            )
        excluded = [f"{row['id']}: {row['excluded']}" for row in candidate.landmarks if row.get("excluded")]
        lines.append("Landmark exclusions: " + ("; ".join(excluded) if excluded else "None"))
        candidate_sections.append((f"Candidate: {candidate.name}", "\n".join(lines)))
    sections.append(("Software versions", "; ".join(f"{key}: {value}" for key, value in analysis.versions.items())))
    return candidate_sections + sections


def _write_pdf_report(
    path: Path,
    title: str,
    sections: list[tuple[str, str]],
    figures: list[tuple[str, Path]],
    *,
    include_paths: bool,
) -> None:
    """Write paginated text and diagnostic figures using the existing Matplotlib dependency."""
    from matplotlib import image as mpl_image
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    privacy = (
        "Source directory paths included." if include_paths else "Source directory paths omitted; filenames retained."
    )
    content = [
        ("Report overview", "Independent accuracy, fitting evidence, region overlap and deformation.\n" + privacy)
    ]
    content.extend(sections)
    content.append(("Methods and interpretation", "Definitions and limits of the reported measurements."))
    content.extend(_METHODS[:-1] if figures else _METHODS)
    page_number = 0
    with PdfPages(
        path, metadata={"Title": title, "Author": "image2image", "Subject": "Registration validation"}
    ) as pdf:

        def new_page(*, landscape: bool = False) -> Figure:
            nonlocal page_number
            page_number += 1
            page = Figure(figsize=(11.69, 8.27) if landscape else (8.27, 11.69), facecolor="white")
            page.text(0.07, 0.955, title, fontsize=12, weight="bold", color="#163f55")
            page.text(0.07, 0.035, "image2image | Registration validation", fontsize=8, color="#526575")
            page.text(0.93, 0.035, str(page_number), ha="right", fontsize=8, color="#526575")
            return page

        page, y = new_page(), 0.90
        for heading, text in content:
            heading_lines = textwrap.wrap(heading, 72) or [""]
            text_lines = [line for paragraph in text.splitlines() for line in (textwrap.wrap(paragraph, 98) or [""])]
            block_height = 0.023 * len(heading_lines) + 0.017 * len(text_lines) + 0.02
            required_height = (
                min(block_height, 0.023 * len(heading_lines) + 0.017 * 3) if block_height > 0.82 else block_height
            )
            if y - required_height < 0.08:
                pdf.savefig(page)
                page, y = new_page(), 0.90
            for line in heading_lines:
                if y < 0.08:
                    pdf.savefig(page)
                    page, y = new_page(), 0.90
                page.text(0.07, y, line, fontsize=11, weight="bold", color="#163f55", parse_math=False)
                y -= 0.023
            for line in text_lines:
                if y < 0.08:
                    pdf.savefig(page)
                    page, y = new_page(), 0.90
                page.text(0.07, y, line, fontsize=9, color="#203040", parse_math=False)
                y -= 0.017
            y -= 0.02
        pdf.savefig(page)
        for heading, image_path in figures:
            page = new_page(landscape=True)
            page.text(0.07, 0.90, "Diagnostic figures", fontsize=13, weight="bold", color="#163f55")
            for index, line in enumerate(textwrap.wrap(heading, 85)[:5]):
                page.text(0.07, 0.87 - index * 0.019, line, fontsize=10, parse_math=False)
            axes = page.add_axes((0.06, 0.14, 0.88, 0.63))
            axes.imshow(mpl_image.imread(image_path))
            axes.set_axis_off()
            for index, line in enumerate(textwrap.wrap(_METHODS[-1][1], 155)):
                page.text(0.07, 0.12 - index * 0.014, line, fontsize=8, color="#203040")
            pdf.savefig(page)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize a preview robustly without changing numerical measurements."""
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, (1, 99)) if len(finite) else (0, 1)
    return np.nan_to_num(np.clip((image - low) / max(float(high - low), 1e-12), 0, 1))


def plot_validation_candidate(
    analysis: ValidationAnalysis,
    result: CandidateAnalysis,
    *,
    error_limit: float | None = None,
    area_limit: float | None = None,
) -> bytes:
    """Render calibrated overlays and point measurements as a standalone PNG."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if result.map_path is None:
        raise ValueError("Candidate maps are unavailable.")
    with np.load(result.map_path) as stored:
        stride = max(1, int(np.ceil(max(analysis.grid.shape) / 1000)))
        arrays = {
            name: stored[name][::stride, ::stride]
            if stored[name].ndim >= 2 and not name.startswith("shift")
            else stored[name]
            for name in stored.files
            if name not in ("origin_xy_um", "spacing_xy_um")
        }
    automatic = "shift_um" in arrays or bool(result.automatic)
    fig = Figure(figsize=(12, 13.5 if automatic else 9), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(3 if automatic else 2, 2)
    grid = analysis.grid
    sx, sy = grid.spacing
    ox, oy = grid.origin
    height, width = grid.shape
    extent = (ox - sx / 2, ox + (width - 0.5) * sx, oy + (height - 0.5) * sy, oy - sy / 2)
    fixed = normalize_image(arrays["fixed_image"])
    moving = normalize_image(arrays["registered_image"])
    axes[0, 0].imshow(np.stack((fixed, moving, fixed), axis=-1), extent=extent)
    axes[0, 0].set_title("Fixed (magenta) / registered moving (green)")
    rows = [row for row in result.landmarks if row["error_um"] is not None and row.get("source") != "automatic"]
    if rows:
        errors = np.array([r["error_um"] for r in rows])
        x, y = np.array([[r["moving_x"], r["moving_y"]] for r in rows]).T
        scatter = axes[0, 1].scatter(
            x, y, c=errors, vmin=0, vmax=error_limit or max(float(errors.max()), 1), cmap="magma", s=35
        )
        axes[0, 1].quiver(
            x,
            y,
            [r["dx_um"] for r in rows],
            [r["dy_um"] for r in rows],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#3274a1",
        )
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_aspect("equal", adjustable="datalim")
        fig.colorbar(scatter, ax=axes[0, 1], label="Validation error (um)")
    else:
        axes[0, 1].text(0.5, 0.5, "No independent validation landmarks", ha="center", transform=axes[0, 1].transAxes)
    axes[0, 1].set_title("Measured landmarks and residuals - moving frame")
    area = arrays["log2_area_ratio"]
    finite_area = np.abs(area[np.isfinite(area)])
    limit = area_limit or (max(float(finite_area.max()), 0.1) if len(finite_area) else 1)
    im = axes[1, 0].imshow(np.ma.masked_invalid(area), cmap="coolwarm", vmin=-limit, vmax=limit, extent=extent)
    flags = np.zeros((*area.shape, 4), dtype=np.float32)
    flags[arrays["nonpositive"]] = (0.5, 0.0, 0.5, 1.0)
    axes[1, 0].imshow(flags, extent=extent)
    axes[1, 0].set_facecolor("#dddddd")
    axes[1, 0].set_title("Local area change; purple = nonpositive Jacobian")
    fig.colorbar(im, ax=axes[1, 0], label="log2 moving-to-fixed area ratio; + expansion / - compression")
    if "fixed_mask" in arrays and analysis.session.mask_pairs:
        disagreement = np.zeros((*fixed.shape, 3), dtype=np.float32)
        for pair in analysis.session.mask_pairs:
            a, b = arrays["fixed_mask"] == pair.fixed_label, arrays["moving_mask"] == pair.moving_label
            disagreement[a & b] = (0.3, 0.8, 0.3)
            disagreement[a & ~b] = (1, 0, 1)
            disagreement[b & ~a] = (0, 0.7, 1)
        axes[1, 1].imshow(disagreement, extent=extent)
    else:
        axes[1, 1].text(
            0.5, 0.5, "Region overlap unavailable; see report notes", ha="center", transform=axes[1, 1].transAxes
        )
    axes[1, 1].set_title("Masks: green agreement, magenta fixed, cyan moving")
    if automatic:
        _plot_automatic(fig, axes[2], arrays, result, fixed, extent)
    for ax in axes.flat:
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")
    fig.suptitle(result.name)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    return buffer.getvalue()


def _plot_automatic(
    fig: Any, axes: Any, arrays: dict[str, np.ndarray], result: CandidateAnalysis, fixed: np.ndarray, extent: tuple
) -> None:
    """Draw the residual shift map and automatic landmark residuals."""
    if "shift_um" in arrays:
        (ox, oy), (sx, sy) = arrays["shift_origin_xy_um"], arrays["shift_spacing_xy_um"]
        height, width = arrays["shift_um"].shape
        tiles = (ox - sx / 2, ox + (width - 0.5) * sx, oy + (height - 0.5) * sy, oy - sy / 2)
        axes[0].imshow(fixed, cmap="gray", extent=extent)
        shown = axes[0].imshow(np.ma.masked_invalid(arrays["shift_um"]), cmap="magma", alpha=0.7, extent=tiles)
        fig.colorbar(shown, ax=axes[0], label="Residual shift (um, fixed frame)")
    else:
        axes[0].text(0.5, 0.5, "Residual shift map not requested", ha="center", transform=axes[0].transAxes)
    axes[0].set_title("Residual shift map; blank tiles rejected")
    rows = [row for row in result.landmarks if row.get("source") == "automatic" and row["error_um"] is not None]
    if rows:
        errors = np.array([r["error_um"] for r in rows])
        x, y = np.array([[r["moving_x"], r["moving_y"]] for r in rows]).T
        scatter = axes[1].scatter(x, y, c=errors, vmin=0, vmax=max(float(errors.max()), 1e-9), cmap="magma", s=12)
        axes[1].invert_yaxis()
        axes[1].set_aspect("equal", adjustable="datalim")
        fig.colorbar(scatter, ax=axes[1], label="Automatic landmark error (um, moving frame)")
    else:
        message = result.automatic.get("reason") or "No automatic landmarks detected"
        axes[1].text(0.5, 0.5, message, ha="center", wrap=True, transform=axes[1].transAxes)
    axes[1].set_title("Automatic landmarks - moving frame")


DIRECTION_SOURCES = {
    "independent": "Independent landmarks (moving frame)",
    "automatic": "Automatic landmarks (moving frame)",
    "shift_tiles": "Residual shift tiles (fixed frame)",
}


def _xy(value: tuple[float, float] | None) -> str | None:
    return None if value is None else f"{value[0]:.4g}, {value[1]:.4g}"


def localization_rows(localization: dict[str, Any]) -> list[dict[str, Any]]:
    """Tabulate localization precision and the registration error budget for independent and automatic evidence.

    Returns an empty list for results analyzed without localization support.
    """
    if not localization:
        return []
    manual, matcher = localization.get("manual") or {}, localization.get("matcher") or {}
    groups = [localization.get("independent") or {}, localization.get("automatic") or {}]
    matcher_text = _xy(matcher.get("sigma_moving_xy")) or matcher.get("reason") or "Not requested"
    manual_used = manual.get("method") or "Not measured"

    def both(key: str) -> list[Any]:
        return [group.get(key) for group in groups]

    rows: list[tuple[str, Any, Any]] = [
        ("Manual click estimate", manual_used, "Not used"),
        ("Fixed click SD XY (um)", _xy(manual.get("sigma_fixed_xy")), "Not used"),
        ("Moving click SD XY (um)", _xy(manual.get("sigma_moving_xy")), "Not used"),
        ("Matcher SD XY (um)", matcher_text, matcher_text),
        (
            "Residuals with a noise estimate",
            *(f"{g.get('count', 0)}/{g.get('count', 0) + g.get('missing', 0)}" for g in groups),
        ),
        ("Observed RMS (um)", *both("observed_rms")),
        ("Localization RMS (um)", *both("fle_rms")),
        ("Corrected RMS (um)", *both("corrected_rms")),
        ("Localization share of squared error", *both("fraction_localization")),
        (
            "Within localization noise",
            *(f"{g['within_noise']}/{g['count']}" if g.get("count") else None for g in groups),
        ),
    ]
    if manual.get("note"):
        rows.append(("Note", manual["note"], ""))
    return [{"statistic": name, "independent": a, "automatic": b} for name, a, b in rows]


def direction_colors(
    dx: np.ndarray, dy: np.ndarray, valid: np.ndarray, limit: float | None = None
) -> tuple[np.ndarray, float]:
    """Color residual vectors by direction (hue) and magnitude (brightness); invalid samples are gray.

    Hue follows ``atan2(dy, dx)`` in image coordinates (Y down), matching :func:`direction_wheel`.
    Brightness saturates at ``limit``, by default the 95th percentile of valid magnitudes.

    Returns
    -------
    tuple
        Float32 RGB array of shape ``dx.shape + (3,)`` and the magnitude limit used.
    """
    from matplotlib.colors import hsv_to_rgb

    dx, dy = np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(dx) & np.isfinite(dy)
    magnitude = np.hypot(np.nan_to_num(dx), np.nan_to_num(dy))
    if limit is None:
        limit = float(np.percentile(magnitude[valid], 95)) if valid.any() else 1.0
    limit = max(limit, 1e-12)
    hsv = np.ones((*dx.shape, 3))
    hsv[..., 0] = (np.arctan2(np.nan_to_num(dy), np.nan_to_num(dx)) / (2 * np.pi)) % 1
    hsv[..., 2] = np.clip(magnitude / limit, 0, 1)
    rgb = hsv_to_rgb(hsv).astype(np.float32)
    rgb[~valid] = 0.5
    return rgb, limit


def direction_wheel(size: int = 96) -> np.ndarray:
    """Return an RGBA color wheel (Y down) for :func:`direction_colors`; outside the disk is transparent."""
    yy, xx = np.mgrid[:size, :size]
    dx, dy = (xx - (size - 1) / 2) / (size / 2), (yy - (size - 1) / 2) / (size / 2)
    inside = np.hypot(dx, dy) <= 1
    rgb, _ = direction_colors(dx, dy, inside, 1.0)
    return np.dstack((rgb, inside.astype(np.float32)))


def direction_text(stats: dict[str, Any]) -> str:
    """Summarize a direction-bias result in one short paragraph."""
    if not stats or stats.get("verdict") == "unavailable":
        return stats.get("reason") or "Directional summary unavailable."
    text = f"Error direction: {stats['verdict']}"
    if stats.get("oblique"):
        text += f" (oblique, principal axis {stats['angle_deg']:.0f} deg)"
    text += f". Basis: {stats['basis']}. Mean bias dX {stats['mean_dx']:.4g} um, dY {stats['mean_dy']:.4g} um"
    if stats.get("bias_p") is not None:
        text += f" (Hotelling p = {stats['bias_p']:.3g})"
    text += f"; RMS dX {stats['rms_dx']:.4g} um, dY {stats['rms_dy']:.4g} um."
    if stats.get("note"):
        text += f" {stats['note']}"
    return text


def plot_direction_bias(target: Any, rose: Any, dx: np.ndarray, dy: np.ndarray, stats: dict[str, Any]) -> None:
    """Draw a residual target plot and a magnitude-weighted rose diagram.

    Parameters
    ----------
    target
        Cartesian Matplotlib axes; X right and Y down, matching image coordinates.
    rose
        Polar Matplotlib axes; angles increase clockwise from +X, so 90 deg points down (+Y).
    dx, dy
        Residual components in um.
    stats
        Output of :func:`image2image.utils.validation.direction_bias`.
    """
    from matplotlib.patches import Circle, Ellipse

    finite = np.isfinite(dx) & np.isfinite(dy)
    dx, dy = np.asarray(dx, dtype=float)[finite], np.asarray(dy, dtype=float)[finite]
    if not len(dx):
        for axes in (target, rose):
            axes.text(0.5, 0.5, "No residuals", ha="center", va="center", transform=axes.transAxes)
        return
    magnitude = np.hypot(dx, dy)
    limit = 1.15 * max(float(magnitude.max()), 1e-9)
    if stats.get("sigma_x"):
        noise = Ellipse(
            (0, 0),
            2 * np.sqrt(CHI2_2D_95) * stats["sigma_x"],
            2 * np.sqrt(CHI2_2D_95) * stats["sigma_y"],
            color="#9ec5d6",
            alpha=0.4,
            label="95% localization noise",
        )
        target.add_patch(noise)
        limit = max(limit, 1.05 * np.sqrt(CHI2_2D_95) * max(stats["sigma_x"], stats["sigma_y"]))
    rms = float(np.sqrt(np.mean(magnitude**2)))
    target.add_patch(Circle((0, 0), rms, fill=False, ls="--", color="#526575", label=f"RMS {rms:.3g} um"))
    target.axhline(0, color="#999999", lw=0.8)
    target.axvline(0, color="#999999", lw=0.8)
    target.scatter(dx, dy, s=16, color="#3274a1", zorder=3)
    if stats.get("covariance") is not None:
        values, axes = np.linalg.eigh(np.asarray(stats["covariance"]))
        angle = float(np.degrees(np.arctan2(axes[1, 1], axes[0, 1])))
        width, height = 2 * np.sqrt(CHI2_2D_95 * np.clip(values[::-1], 0, None))
        target.add_patch(
            Ellipse(
                (stats["mean_dx"], stats["mean_dy"]),
                width,
                height,
                angle=angle,
                fill=False,
                color="#c44e52",
                lw=1.5,
                label="95% covariance",
            )
        )
        target.annotate(
            "",
            xy=(stats["mean_dx"], stats["mean_dy"]),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "#c44e52", "lw": 2},
        )
    target.set_xlim(-limit, limit)
    target.set_ylim(limit, -limit)
    target.set_aspect("equal")
    target.set_xlabel("dX (um)")
    target.set_ylabel("dY (um, down)")
    target.legend(loc="upper right", fontsize=7)
    target.set_title(f"Residuals: {stats.get('verdict', 'unavailable')}", fontsize=10)
    bins = np.linspace(-np.pi, np.pi, 17)
    counts, _ = np.histogram(np.arctan2(dy, dx), bins=bins, weights=magnitude)
    rose.bar(bins[:-1], counts, width=np.diff(bins), align="edge", color="#3274a1", alpha=0.75, edgecolor="white")
    rose.set_theta_zero_location("E")
    rose.set_theta_direction(-1)
    rose.set_yticklabels([])
    rose.set_title("Direction, weighted by magnitude", fontsize=10)


def plot_direction_figure(result: CandidateAnalysis) -> bytes | None:
    """Render every available direction-bias source of a candidate as a PNG, or ``None`` if none.

    The residual shift tile row also shows the residual direction map with its color wheel.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    arrays = None
    if result.map_path is not None and Path(result.map_path).exists():
        with np.load(result.map_path) as stored:
            arrays = {key: stored[key] for key in stored.files if key.startswith("shift_")}
    sources = [
        key for key in DIRECTION_SOURCES if result.direction.get(key, {}).get("verdict", "unavailable") != "unavailable"
    ]
    if not sources:
        return None
    columns = 3 if "shift_tiles" in sources else 2
    fig = Figure(figsize=(5 * columns, 4.6 * len(sources)), constrained_layout=True)
    FigureCanvasAgg(fig)
    for row, source in enumerate(sources):
        target = fig.add_subplot(len(sources), columns, columns * row + 1)
        rose = fig.add_subplot(len(sources), columns, columns * row + 2, projection="polar")
        dx, dy = direction_vectors(result, source, arrays)
        plot_direction_bias(target, rose, dx, dy, result.direction[source])
        target.set_title(f"{DIRECTION_SOURCES[source]}: {result.direction[source]['verdict']}", fontsize=10)
        if source == "shift_tiles":
            _plot_direction_map(fig.add_subplot(len(sources), columns, columns * row + 3), arrays)
    fig.suptitle(f"Directional bias - {result.name}")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    return buffer.getvalue()


def _plot_direction_map(axes: Any, arrays: dict[str, np.ndarray]) -> None:
    """Draw the residual direction map in physical fixed-frame coordinates with an inset color wheel."""
    rgb, limit = direction_colors(arrays["shift_x_um"], arrays["shift_y_um"], arrays["shift_valid"])
    (ox, oy), (sx, sy) = arrays["shift_origin_xy_um"], arrays["shift_spacing_xy_um"]
    height, width = rgb.shape[:2]
    axes.imshow(rgb, extent=(ox - sx / 2, ox + (width - 0.5) * sx, oy + (height - 0.5) * sy, oy - sy / 2))
    axes.set_title(f"Residual direction map; full brightness at {limit:.3g} um", fontsize=10)
    axes.set_xlabel("X (um)")
    axes.set_ylabel("Y (um)")
    inset = axes.inset_axes((0.78, 0.78, 0.2, 0.2))
    inset.imshow(direction_wheel())
    inset.set_axis_off()


def export_validation_report(
    analysis: ValidationAnalysis, output_dir: str | Path, *, include_paths: bool = True
) -> Path:
    """Export HTML, PDF and measurement data; optionally retain filenames without source directories."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a new or empty report directory to avoid overwriting an earlier report.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not include_paths:
        analysis = _report_snapshot(analysis, _source_paths(analysis.session.model_dump(mode="json")))
    with TemporaryDirectory(prefix=".i2validate-report-", dir=output_dir.parent) as temporary:
        _write_report(analysis, Path(temporary), include_paths=include_paths)
        if output_dir.exists():
            output_dir.rmdir()  # Fails safely if another writer populated this directory.
        Path(temporary).replace(output_dir)
    return output_dir / "report.html"


def export_validation_study_report(
    analysis: StudyAnalysis, output_dir: str | Path, *, include_paths: bool = True
) -> Path:
    """Export HTML, PDF and separate pair measurements, optionally omitting source directories."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a new or empty report directory to avoid overwriting an earlier report.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    study = analysis.study.model_dump(mode="json")
    sources = _source_paths(study)
    for pair in analysis.pairs:
        if pair.analysis is not None:
            sources += _source_paths(pair.analysis.session.model_dump(mode="json"))
    sources = tuple(sorted(set(sources), key=len, reverse=True))
    if not include_paths:
        study = _redact_paths(study, sources)
    with TemporaryDirectory(prefix=".i2validate-report-", dir=output_dir.parent) as temporary:
        root = Path(temporary)
        summaries, landmarks, masks, records, sections = [], [], [], [], []
        pdf_sections, figures = [], []
        for index, pair in enumerate(analysis.pairs, 1):
            if not include_paths:
                pair = replace(
                    pair,
                    pair_id=_redact_paths(pair.pair_id, sources),
                    name=_redact_paths(pair.name, sources),
                    error=_redact_paths(pair.error, sources),
                    analysis=_report_snapshot(pair.analysis, sources) if pair.analysis is not None else None,
                )
            identity = {"pair_id": pair.pair_id, "moving_image": pair.name}
            status = "complete" if pair.analysis is not None else "failed" if pair.error else "not_analyzed"
            record = {**identity, "status": status, "error": pair.error, "analysis": None}
            sections.append(f"<section><h2>{html.escape(pair.name)}</h2>")
            if pair.analysis is None:
                summaries.append({**identity, "candidate": "", "status": status, "error": pair.error})
                message = (
                    f"Analysis failed: {pair.error}" if pair.error else "Not analyzed. No measurements are available."
                )
                sections.append(f"<p>{html.escape(message)}</p>")
                pdf_sections.append((f"Image pair: {pair.name}", message))
            else:
                directory = root / f"pair-{index}"
                directory.mkdir()
                _write_report(pair.analysis, directory, include_paths=include_paths)
                payload = json.loads((directory / "results.json").read_text())
                for candidate in payload["candidates"]:
                    if candidate["map_path"]:
                        candidate["map_path"] = f"pair-{index}/{candidate['map_path']}"
                record["analysis"] = payload
                rows = comparison_rows(pair.analysis)
                summaries.extend({**identity, **row, "error": ""} for row in rows)
                sections.append(
                    f"<p><strong>Analyzed:</strong> {html.escape(pair.analysis.created)}<br>"
                    f"<strong>Validation independence:</strong> {html.escape(pair.analysis.session.independence_note)}</p>"
                    + _comparison_table(rows)
                    + _context(pair.analysis)
                )
                pdf_sections.append((f"Image pair: {pair.name}", "Measurements are separate for this pair."))
                pdf_sections.extend(_pdf_pair_sections(pair.analysis))
                for candidate_index, candidate in enumerate(pair.analysis.candidates, 1):
                    landmarks.extend({**identity, "candidate": candidate.name, **row} for row in candidate.landmarks)
                    masks.extend({**identity, "candidate": candidate.name, **row} for row in candidate.masks)
                    png = (directory / f"candidate-{candidate_index}.png").read_bytes()
                    direction_path = directory / f"candidate-{candidate_index}-direction.png"
                    direction_png = direction_path.read_bytes() if direction_path.exists() else None
                    sections.append(_candidate_section(candidate, png, level=3, direction_png=direction_png))
                    figures.append((f"{pair.name}: {candidate.name}", directory / f"candidate-{candidate_index}.png"))
                    if direction_png is not None:
                        figures.append((f"{pair.name}: {candidate.name} - directional bias", direction_path))
                sections.append(
                    f"<p><a href='pair-{index}/report.html'>Full pair report</a> | "
                    f"<a href='pair-{index}/report.pdf'>Pair PDF report</a></p>"
                )
                provenance = {key: value for key, value in payload.items() if key != "candidates"}
                sections.append(
                    "<details><summary>Inputs, calibration, grid and software versions</summary><pre>"
                    + html.escape(json.dumps(provenance, indent=2))
                    + "</pre></details>"
                )
            sections.append("</section>")
            records.append(record)
        for filename, rows in (("summary.csv", summaries), ("landmarks.csv", landmarks), ("masks.csv", masks)):
            columns = list(dict.fromkeys(key for row in rows for key in row)) or ["pair_id", "moving_image"]
            _write_csv(root / filename, rows, columns)
        payload = {"schema_version": "2.0", "study": study, "pairs": records}
        (root / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (root / "session.i2validate.json").write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        overview = [
            {"moving_image": record["moving_image"], "status": record["status"], "error": record["error"]}
            for record in records
        ]
        body = "<h2>Study overview</h2><p>Each moving image is evaluated against the shared fixed reference. "
        body += "Measurements are not pooled across image pairs.</p>" + _table(overview) + "".join(sections)
        document = _document("Registration validation study", body, include_paths)
        (root / "report.html").write_text(document, encoding="utf-8")
        _write_pdf_report(
            root / "report.pdf", "Registration validation study", pdf_sections, figures, include_paths=include_paths
        )
        if output_dir.exists():
            output_dir.rmdir()
        root.replace(output_dir)
    return output_dir / "report.html"


def _write_report(analysis: ValidationAnalysis, output_dir: Path, *, include_paths: bool) -> Path:
    summaries = comparison_rows(analysis)
    _write_csv(output_dir / "summary.csv", summaries)
    _write_csv(
        output_dir / "landmarks.csv", [{"candidate": c.name, **row} for c in analysis.candidates for row in c.landmarks]
    )
    _write_csv(output_dir / "masks.csv", [{"candidate": c.name, **row} for c in analysis.candidates for row in c.masks])
    payload = {
        "schema_version": "1.0",
        "created": analysis.created,
        "versions": analysis.versions,
        "session": analysis.session.model_dump(mode="json"),
        "grid": analysis.grid.model_dump(),
        "candidates": [],
    }
    error_limit = max(
        [
            row["error_um"]
            for c in analysis.candidates
            for row in c.landmarks
            if row["error_um"] is not None and row.get("source") != "automatic"
        ]
        or [1]
    )
    area_limit = 0.1
    for candidate in analysis.candidates:
        if candidate.map_path is not None:
            with np.load(candidate.map_path) as arrays:
                values = np.abs(arrays["log2_area_ratio"])
                values = values[np.isfinite(values)]
                if values.size:
                    area_limit = max(area_limit, float(values.max()))
    sections, figures = [], []
    for index, candidate in enumerate(analysis.candidates, 1):
        record = asdict(candidate)
        record["map_path"] = f"candidate-{index}.npz" if candidate.map_path else None
        payload["candidates"].append(record)
        if candidate.map_path:
            shutil.copyfile(candidate.map_path, output_dir / record["map_path"])
        png = plot_validation_candidate(analysis, candidate, error_limit=max(error_limit, 1e-9), area_limit=area_limit)
        (output_dir / f"candidate-{index}.png").write_bytes(png)
        direction_png = plot_direction_figure(candidate)
        sections.append(_candidate_section(candidate, png, direction_png=direction_png))
        figures.append((candidate.name, output_dir / f"candidate-{index}.png"))
        if direction_png is not None:
            (output_dir / f"candidate-{index}-direction.png").write_bytes(direction_png)
            figures.append((f"{candidate.name} - directional bias", output_dir / f"candidate-{index}-direction.png"))
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "session.i2validate.json").write_text(
        analysis.session.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    body = f"<h2>Comparison summary</h2><p><strong>Analyzed:</strong> {html.escape(analysis.created)}<br>"
    body += f"<strong>Validation independence:</strong> {html.escape(analysis.session.independence_note)}</p>"
    body += _comparison_table(summaries) + _context(analysis) + "".join(sections)
    body += "<details><summary>Full provenance and software versions</summary><pre>"
    body += html.escape(json.dumps({key: value for key, value in payload.items() if key != "candidates"}, indent=2))
    body += "</pre></details>"
    text = _document("Registration validation", body, include_paths)
    path = output_dir / "report.html"
    path.write_text(text, encoding="utf-8")
    _write_pdf_report(
        output_dir / "report.pdf",
        "Registration validation",
        _pdf_pair_sections(analysis),
        figures,
        include_paths=include_paths,
    )
    return path
