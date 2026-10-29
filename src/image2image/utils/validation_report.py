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
        "Automatic landmarks are SIFT correspondences between the fixed and registered images, refined by "
        "normalized cross-correlation at native resolution; residuals use the independent-landmark definition "
        "G(fixed) - moving in the moving frame, and outliers above the median plus 3.5 robust standard deviations "
        "are flagged. Both depend on image similarity and can fail or be biased for different stains or "
        "modalities, repetitive texture, or tissue damage; too few reliable measurements are reported as "
        "unavailable, not as zero error.",
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
            "automatic_median_um": result.automatic.get("median"),
            "residual_map_valid_fraction": result.residual_map.get("valid_fraction"),
            "residual_map_median_um": result.residual_map.get("median"),
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


def _candidate_section(candidate: CandidateAnalysis, png: bytes, level: int = 2) -> str:
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
        + _automatic_section(candidate, heading)
        + "<details><summary>Per-landmark residuals and exclusions</summary>"
        f"{_table(candidate.landmarks)}</details></section>"
    )


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
        if candidate.automatic_landmarks:
            text += f"<details><summary>Automatic landmark residuals</summary>{_table(candidate.automatic_landmarks)}</details>"
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
    automatic = "shift_um" in arrays or bool(result.automatic_landmarks)
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
    rows = [row for row in result.landmarks if row["error_um"] is not None]
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
    rows = [row for row in result.automatic_landmarks if row["error_um"] is not None]
    if rows:
        errors = np.array([r["error_um"] for r in rows])
        x, y = np.array([[r["moving_x"], r["moving_y"]] for r in rows]).T
        limit = float(np.max([r["error_um"] for r in rows if not r["outlier"]] or [errors.max()]))
        scatter = axes[1].scatter(x, y, c=errors, vmin=0, vmax=max(limit, 1e-9), cmap="magma", s=12)
        axes[1].invert_yaxis()
        axes[1].set_aspect("equal", adjustable="datalim")
        fig.colorbar(scatter, ax=axes[1], label="Automatic landmark error (um, moving frame)")
    else:
        message = result.automatic.get("reason") or "Automatic landmarks not requested"
        axes[1].text(0.5, 0.5, message, ha="center", wrap=True, transform=axes[1].transAxes)
    axes[1].set_title("Automatic landmarks - moving frame")


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
        summaries, landmarks, masks, automatic, records, sections = [], [], [], [], [], []
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
                    automatic.extend(
                        {**identity, "candidate": candidate.name, **row} for row in candidate.automatic_landmarks
                    )
                    png = (directory / f"candidate-{candidate_index}.png").read_bytes()
                    sections.append(_candidate_section(candidate, png, level=3))
                    figures.append((f"{pair.name}: {candidate.name}", directory / f"candidate-{candidate_index}.png"))
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
        tables = [("summary.csv", summaries), ("landmarks.csv", landmarks), ("masks.csv", masks)]
        if automatic:
            tables.append(("automatic_landmarks.csv", automatic))
        for filename, rows in tables:
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
    automatic = [{"candidate": c.name, **row} for c in analysis.candidates for row in c.automatic_landmarks]
    if automatic:
        _write_csv(output_dir / "automatic_landmarks.csv", automatic)
    payload = {
        "schema_version": "1.0",
        "created": analysis.created,
        "versions": analysis.versions,
        "session": analysis.session.model_dump(mode="json"),
        "grid": analysis.grid.model_dump(),
        "candidates": [],
    }
    error_limit = max(
        [row["error_um"] for c in analysis.candidates for row in c.landmarks if row["error_um"] is not None] or [1]
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
        sections.append(_candidate_section(candidate, png))
        figures.append((candidate.name, output_dir / f"candidate-{index}.png"))
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
