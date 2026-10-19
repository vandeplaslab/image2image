"""Self-contained visual reports and machine-readable validation exports."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from image2image.models.validation import CandidateAnalysis, ValidationAnalysis


def comparison_rows(analysis: ValidationAnalysis) -> list[dict[str, Any]]:
    """Return comparison rows with explicit metric names, units, and denominators."""
    return [
        {
            "candidate": result.name,
            "status": result.status,
            "validation_count": result.validation.get("count", 0),
            "validation_eligible": result.validation.get("eligible", 0),
            **{
                f"validation_{key}_um": result.validation.get(key)
                for key in ("mean", "median", "rms", "p95", "maximum")
            },
            "fitting_rms_um": result.fitting.get("rms"),
            "coverage": result.deformation.get("coverage"),
            "nonpositive_fraction": result.deformation.get("nonpositive_fraction"),
        }
        for result in analysis.candidates
    ]


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>Unavailable: no evidence supplied.</p>"
    keys = list(rows[0])
    headings = "".join(f"<th>{html.escape(key)}</th>" for key in keys)
    body = []
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key)
            text = "Unavailable" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table'><table><thead><tr>{headings}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or (list(rows[0]) if rows else ["candidate"]))
        writer.writeheader()
        writer.writerows(rows)


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
            for name in stored.files
            if name not in ("origin_xy_um", "spacing_xy_um")
        }
    fig = Figure(figsize=(12, 9), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
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
        axes[1, 1].text(0.5, 0.5, "No corresponding masks supplied", ha="center", transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Masks: green agreement, magenta fixed, cyan moving")
    for ax in axes.flat:
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")
    fig.suptitle(result.name)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    return buffer.getvalue()


def export_validation_report(analysis: ValidationAnalysis, output_dir: str | Path) -> Path:
    """Export HTML, CSV, JSON, PNG and NPZ data into a new, empty report directory."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a new or empty report directory to avoid overwriting an earlier report.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".i2validate-report-", dir=output_dir.parent) as temporary:
        _write_report(analysis, Path(temporary))
        if output_dir.exists():
            output_dir.rmdir()  # Fails safely if another writer populated this directory.
        Path(temporary).replace(output_dir)
    return output_dir / "report.html"


def _write_report(analysis: ValidationAnalysis, output_dir: Path) -> Path:
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
    sections = []
    for index, candidate in enumerate(analysis.candidates, 1):
        record = asdict(candidate)
        record["map_path"] = f"candidate-{index}.npz" if candidate.map_path else None
        payload["candidates"].append(record)
        if candidate.map_path:
            shutil.copyfile(candidate.map_path, output_dir / record["map_path"])
        png = plot_validation_candidate(analysis, candidate, error_limit=max(error_limit, 1e-9), area_limit=area_limit)
        (output_dir / f"candidate-{index}.png").write_bytes(png)
        sections.append(
            f"<section><h2>{html.escape(candidate.name)}</h2>"
            f"<img alt='Registration diagnostics' src='data:image/png;base64,{base64.b64encode(png).decode()}'>"
            f"<h3>Fitting evidence (um, moving frame)</h3>{_table([candidate.fitting])}"
            f"<h3>Mask overlap on fixed grid</h3>{_table(candidate.masks)}"
            f"<h3>Deformation</h3>{_table([candidate.deformation])}"
            f"<details><summary>Per-landmark validation and exclusions</summary>"
            f"{_table(candidate.landmarks)}</details></section>"
        )
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (output_dir / "session.i2validate.json").write_text(analysis.session.model_dump_json(indent=2) + "\n")
    text = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Registration validation</title>
<style>body{font:15px system-ui,sans-serif;max-width:1400px;margin:36px auto;padding:0 24px;color:#203040}
h1,h2{color:#163f55}section{border-top:1px solid #ccc;margin-top:30px}img{width:100%;height:auto}
.table{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border:1px solid #ddd;
text-align:left}th{background:#edf3f6}details{margin:20px 0}p{line-height:1.6}</style><h1>Registration validation</h1>
<p>Validation residuals are G(fixed) - moving, in moving-image physical coordinates (um).
Fitting residuals describe the landmarks used during registration; they are not independent validation accuracy.
Known fitting landmarks are excluded from every candidate's validation set. Other independence is user-declared.
Excluded rows and evaluated counts are reported explicitly; sparse landmark measurements do not establish accuracy
everywhere. No universal pass/fail threshold is applied.</p>
<p>The signed Jacobian describes the complete fixed-to-moving map, including global affine scaling.
For positive determinants, area ratio = 1 / determinant. The displayed log2 ratio is positive for expansion and
negative for compression when mapping moving tissue to fixed space. Purple identifies orientation reversal or
singularity, which can include intended reflections. Gray identifies unavailable regions and derivative boundaries.
Mask scores are measured on the shared fixed-image grid; moving content outside this field is clipped.
Source coverage is reported. Both-empty masks are unavailable, not perfect overlap.</p>"""
    text += f"<p>Created {html.escape(analysis.created)}. Analysis grid: {analysis.grid.shape}; "
    text += f"spacing XY {analysis.grid.spacing} um. Image fingerprints use file size and modification time.</p>"
    text += f"<p>{html.escape(analysis.session.independence_note)}</p>" + _table(summaries) + "".join(sections)
    text += "<details><summary>Inputs, settings and software versions</summary><pre>"
    text += html.escape(json.dumps({key: value for key, value in payload.items() if key != "candidates"}, indent=2))
    text += "</pre></details></html>"
    path = output_dir / "report.html"
    path.write_text(text, encoding="utf-8")
    return path
