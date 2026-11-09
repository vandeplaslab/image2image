"""Localization (click) error estimates and the registration error budget."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from image2image.models.validation import (
    ImageGeometry,
    LandmarkRepeat,
    ValidationCandidate,
    ValidationDecoy,
    ValidationImage,
    ValidationLandmark,
    ValidationSession,
)
from image2image.utils.validation import ValidationTransform
from image2image.utils.validation_localization import (
    decoy_error,
    localization_correction,
    matcher_precision,
    repeat_spread,
    session_localization,
)

IDENTITY = ValidationTransform(ValidationCandidate(name="identity", kind="aligned", path=""), matrix=np.eye(3))


def _rows(rng: np.random.Generator, n: int, registration: float, fixed: float, moving: float) -> list[dict]:
    """Residual rows for an identity map: registration error plus fixed and moving click noise per axis."""
    residual = rng.normal(0, registration, (n, 2)) + rng.normal(0, fixed, (n, 2)) + rng.normal(0, moving, (n, 2))
    return [
        {
            "id": str(i),
            "fixed_x": 10.0 * i,
            "fixed_y": 5.0,
            "dx_um": float(dx),
            "dy_um": float(dy),
            "error_um": float(np.hypot(dx, dy)),
            "source": "manual",
        }
        for i, (dx, dy) in enumerate(residual)
    ]


def test_repeat_spread_recovers_click_precision() -> None:
    rng = np.random.default_rng(0)
    landmarks = []
    for _ in range(300):
        fixed, moving = rng.uniform(0, 1000, 2), rng.uniform(0, 1000, 2)
        clicks = [(fixed + rng.normal(0, 0.5, 2), moving + rng.normal(0, 1.0, 2)) for _ in range(4)]
        landmarks.append(
            ValidationLandmark(
                fixed=tuple(clicks[0][0]),
                moving=tuple(clicks[0][1]),
                repeats=[LandmarkRepeat(fixed=tuple(f), moving=tuple(m)) for f, m in clicks[1:]],
            )
        )
    landmarks.append(ValidationLandmark(fixed=(0, 0), moving=(0, 0)))  # not repeated: ignored
    result = repeat_spread(landmarks)
    assert result["landmarks"] == 300 and result["clicks"] == 1200
    assert np.allclose(result["sigma_fixed_xy"], 0.5, rtol=0.1)
    assert np.allclose(result["sigma_moving_xy"], 1.0, rtol=0.1)
    assert repeat_spread([ValidationLandmark(fixed=(0, 0), moving=(0, 0))]) == {}


def test_decoy_error_and_session_fallback() -> None:
    rng = np.random.default_rng(1)
    decoys = []
    for _ in range(400):
        point, offset = rng.uniform(0, 100, 2), rng.uniform(-20, 20, 2)
        clicked = point + offset + rng.normal(0, 0.8, 2) + (0.3, 0)
        decoys.append(ValidationDecoy(image="fixed", point=tuple(point), offset=tuple(offset), clicked=tuple(clicked)))
    decoys.append(ValidationDecoy(image="moving", point=(1, 1), offset=(2, 2)))  # not completed
    result = decoy_error(decoys)
    assert result["n_fixed"] == 400 and result["n_moving"] == 0 and result["sigma_moving_xy"] is None
    assert np.allclose(result["sigma_fixed_xy"], (np.hypot(0.8, 0.3), 0.8), rtol=0.1)
    assert result["bias_fixed_xy"][0] == pytest.approx(0.3, abs=0.12)
    session = ValidationSession(
        fixed=ValidationImage(path="f", geometry=ImageGeometry(shape=(10, 10), spacing=(0.5, 0.5))),
        moving=ValidationImage(path="m", geometry=ImageGeometry(shape=(10, 10), spacing=(2.0, 2.0))),
        decoys=decoys,
    )
    chosen = session_localization(session)
    assert chosen["method"] == "decoy" and "pixels" in chosen["note"]
    # Same precision in pixels: four times larger in um on the coarser moving image.
    assert np.allclose(chosen["sigma_moving_xy"], 4 * np.asarray(chosen["sigma_fixed_xy"]))
    assert session_localization(ValidationSession()) == {}


def test_localization_correction_recovers_registration_error() -> None:
    rng = np.random.default_rng(2)
    rows = _rows(rng, 4000, registration=2.0, fixed=0.5, moving=1.0)
    manual = {"sigma_fixed_xy": (0.5, 0.5), "sigma_moving_xy": (1.0, 1.0)}
    summary = localization_correction(rows, IDENTITY, manual, {})["independent"]
    assert summary["count"] == 4000 and summary["missing"] == 0
    assert summary["fle_rms"] == pytest.approx(np.sqrt(2 * (0.25 + 1.0)))
    assert summary["observed_rms"] == pytest.approx(np.sqrt(2 * 5.25), rel=0.05)
    assert summary["corrected_rms"] == pytest.approx(np.sqrt(2 * 4.0), rel=0.05)
    assert summary["fraction_localization"] == pytest.approx(1.25 / 5.25, rel=0.1)
    assert np.allclose(summary["sigma_xy"], np.sqrt(1.25))
    assert rows[0]["fle_um"] == pytest.approx(np.sqrt(2.5))


def test_within_noise_rate_under_pure_click_noise() -> None:
    rng = np.random.default_rng(3)
    rows = _rows(rng, 4000, registration=0.0, fixed=0.5, moving=1.0)
    manual = {"sigma_fixed_xy": (0.5, 0.5), "sigma_moving_xy": (1.0, 1.0)}
    summary = localization_correction(rows, IDENTITY, manual, {})["independent"]
    assert summary["within_noise"] / summary["count"] == pytest.approx(0.95, abs=0.015)
    assert summary["corrected_rms"] < 0.3


def test_localization_correction_sources_and_missing_estimates() -> None:
    rows = [
        {"fixed_x": 0, "fixed_y": 0, "dx_um": 1.0, "dy_um": 0.0, "error_um": 1.0, "source": "manual"},
        {"fixed_x": 0, "fixed_y": 0, "dx_um": 1.0, "dy_um": 0.0, "error_um": 1.0, "source": "assisted"},
        {"fixed_x": 0, "fixed_y": 0, "dx_um": 1.0, "dy_um": 0.0, "error_um": 1.0, "source": "automatic"},
        {"fixed_x": 0, "fixed_y": 0, "dx_um": None, "dy_um": None, "error_um": None, "source": "manual"},
    ]
    manual = {"sigma_fixed_xy": (1.0, 1.0), "sigma_moving_xy": (1.0, 1.0)}
    result = localization_correction(rows, IDENTITY, manual, {})
    assert result["independent"]["count"] == 1 and result["independent"]["missing"] == 1
    assert result["automatic"] == {"count": 0, "missing": 1}
    assert rows[3]["fle_um"] is None and rows[3]["within_noise"] is None
    result = localization_correction(rows, IDENTITY, manual, {"sigma_moving_xy": (0.2, 0.2)})
    assert rows[1]["fle_um"] == pytest.approx(np.sqrt(2 + 0.08))
    assert rows[2]["fle_um"] == pytest.approx(np.sqrt(0.08))
    assert result["automatic"]["within_noise"] == 0  # 1 um residual is far beyond 0.2 um matcher noise


def test_matcher_precision_on_textured_image() -> None:
    rng = np.random.default_rng(4)
    from scipy import ndimage

    image = ndimage.gaussian_filter(rng.normal(size=(256, 256)), 2).astype(np.float32)
    geometry = ImageGeometry(shape=image.shape)
    points = [(80.0, 90.0), (150.0, 120.0), (170.0, 170.0)]
    result = matcher_precision(image, geometry, image, geometry, IDENTITY, points, 32.0, 8.0, 6)
    assert result["points"] == 3 and result["trials"] == 18
    assert max(result["sigma_moving_xy"]) < 0.2
    flat = np.ones((64, 64), dtype=np.float32)
    empty = matcher_precision(
        flat, ImageGeometry(shape=flat.shape), flat, ImageGeometry(shape=flat.shape), IDENTITY, [(32.0, 32.0)], 16, 4, 3
    )
    assert empty["sigma_moving_xy"] is None and empty["rejected"] == 3 and empty["reason"]


def test_analysis_reports_localization(tmp_path: Path) -> None:
    from image2image.utils.validation import analyze_validation_session
    from test_validation import make_validation_session

    session = make_validation_session(tmp_path)
    rng = np.random.default_rng(5)
    session.landmarks = [
        point.model_copy(
            update={
                "repeats": [
                    LandmarkRepeat(
                        fixed=tuple(np.add(point.fixed, rng.normal(0, 0.3, 2))),
                        moving=tuple(np.add(point.moving, rng.normal(0, 0.3, 2))),
                    )
                    for _ in range(2)
                ]
            }
        )
        for point in session.landmarks
    ]
    analysis = analyze_validation_session(session, tmp_path / "maps")
    result = analysis.candidates[0]
    assert result.localization["manual"]["method"] == "re-click"
    measured = [row for row in result.landmarks if row["error_um"] is not None and row["source"] != "automatic"]
    assert all(row["fle_um"] is not None for row in measured)
    assert result.localization["independent"]["count"] == len(measured)
    if result.direction["independent"]["verdict"] != "unavailable":
        assert "localization noise" in result.direction["independent"]["basis"]


def test_report_includes_localization(tmp_path: Path) -> None:
    from image2image.utils.validation import analyze_validation_session
    from image2image.utils.validation_report import export_validation_report, localization_rows
    from test_validation import make_validation_session

    session = make_validation_session(tmp_path)
    session.decoys = [
        ValidationDecoy(image=image, point=(10, 10), offset=(2, 3), clicked=(12 + d, 13 - d))
        for image in ("fixed", "moving")
        for d in (0.4, -0.4)
    ]
    analysis = analyze_validation_session(session, tmp_path / "maps")
    rows = localization_rows(analysis.candidates[0].localization)
    assert rows[0] == {"statistic": "Manual click estimate", "independent": "decoy", "automatic": "Not used"}
    assert localization_rows({}) == []
    report = export_validation_report(analysis, tmp_path / "report")
    text = report.read_text()
    assert "Localization error and corrected accuracy" in text and "Corrected RMS (um)" in text
    assert "corrected_rms_um" in (report.parent / "summary.csv").read_text()
    assert "fle_um" in (report.parent / "landmarks.csv").read_text()
