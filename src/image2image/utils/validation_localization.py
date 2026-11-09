"""Localization (click) error estimates and their separation from registration error.

An observed landmark residual mixes registration error with the error of locating the landmark in each image
(fiducial localization error, FLE). For independent errors the expected squared residual is approximately
``registration^2 + trace(J S_f J^T + S_m)``, where ``S_f`` and ``S_m`` are per-axis localization covariances in the
fixed and moving images and ``J`` is the local Jacobian of the fixed-to-moving map. Localization error is measured
directly, by blind repeat clicks, by decoy trials with a known answer, or by jittering the template matcher, and
then removed from the observed error budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from threading import Event
from typing import Any

import numpy as np

from image2image.models.validation import ImageGeometry, ValidationDecoy, ValidationLandmark, ValidationSession
from image2image.utils.validation import CHI2_2D_95, ValidationTransform, check_cancel

MIN_DECOYS = 2


def _pooled_sd(groups: Sequence[np.ndarray]) -> tuple[float, float] | None:
    """Pooled within-group per-axis standard deviation of XY samples; ``None`` without replicates."""
    squares, dof = np.zeros(2), 0
    for group in groups:
        group = np.asarray(group, dtype=float).reshape(-1, 2)
        if len(group) >= 2:
            squares += ((group - group.mean(axis=0)) ** 2).sum(axis=0)
            dof += len(group) - 1
    if not dof:
        return None
    sd = np.sqrt(squares / dof)
    return float(sd[0]), float(sd[1])


def repeat_spread(landmarks: Sequence[ValidationLandmark]) -> dict[str, Any]:
    """Estimate per-click localization precision from blind repeat clicks.

    Each landmark with repeats contributes its original click and every repeat; the per-axis standard deviation
    is pooled over landmarks. Returns an empty dict when no landmark has been repeated.
    """
    repeated = [point for point in landmarks if point.repeats]
    if not repeated:
        return {}
    fixed = _pooled_sd([[point.fixed, *(r.fixed for r in point.repeats)] for point in repeated])
    moving = _pooled_sd([[point.moving, *(r.moving for r in point.repeats)] for point in repeated])
    return {
        "method": "re-click",
        "sigma_fixed_xy": fixed,
        "sigma_moving_xy": moving,
        "landmarks": len(repeated),
        "clicks": sum(len(point.repeats) + 1 for point in repeated),
    }


def decoy_error(decoys: Sequence[ValidationDecoy]) -> dict[str, Any]:
    """Estimate localization error from completed decoy trials, per image.

    The error of each trial is ``clicked - (point + offset)``; per-axis RMS includes any systematic bias. Images
    with fewer than ``MIN_DECOYS`` completed trials report ``None``. Because a decoy pairs an image with a copy of
    itself, there is no cross-modal ambiguity and the estimate is a lower bound.
    """
    result: dict[str, Any] = {}
    done = [decoy for decoy in decoys if decoy.clicked is not None]
    if not done:
        return result
    result["method"] = "decoy"
    for image in ("fixed", "moving"):
        errors = np.array(
            [np.subtract(d.clicked, np.add(d.point, d.offset)) for d in done if d.image == image], dtype=float
        ).reshape(-1, 2)
        result[f"n_{image}"] = len(errors)
        if len(errors) >= MIN_DECOYS:
            result[f"sigma_{image}_xy"] = tuple(float(v) for v in np.sqrt(np.mean(errors**2, axis=0)))
            result[f"bias_{image}_xy"] = tuple(float(v) for v in errors.mean(axis=0))
        else:
            result[f"sigma_{image}_xy"] = None
    return result


def _scale_sigma(sigma: tuple[float, float], source: ImageGeometry, target: ImageGeometry) -> tuple[float, float]:
    """Assume equal precision in pixels when only one image was measured."""
    return tuple(float(s / a * b) for s, a, b in zip(sigma, source.spacing, target.spacing, strict=True))


def session_localization(session: ValidationSession) -> dict[str, Any]:
    """Choose the manual localization estimate: repeat clicks when available, otherwise decoys.

    When only one image has a measurement, the other image is assumed to have the same precision in pixels;
    this is noted in the result. Returns an empty dict when no estimate is available.
    """
    estimates = [estimate for estimate in (repeat_spread(session.landmarks), decoy_error(session.decoys)) if estimate]
    for estimate in estimates:
        fixed, moving = estimate.get("sigma_fixed_xy"), estimate.get("sigma_moving_xy")
        if fixed is None and moving is None:
            continue
        result = dict(estimate)
        if session.fixed is not None and session.moving is not None:
            if fixed is None:
                result["sigma_fixed_xy"] = _scale_sigma(moving, session.moving.geometry, session.fixed.geometry)
                result["note"] = "Fixed-image precision assumed equal to the moving image in pixels."
            elif moving is None:
                result["sigma_moving_xy"] = _scale_sigma(fixed, session.fixed.geometry, session.moving.geometry)
                result["note"] = "Moving-image precision assumed equal to the fixed image in pixels."
        if result.get("sigma_fixed_xy") is not None and result.get("sigma_moving_xy") is not None:
            return result
    return {}


def matcher_precision(
    fixed_array: Any,
    fixed_geometry: ImageGeometry,
    moving_array: Any,
    moving_geometry: ImageGeometry,
    transform: ValidationTransform,
    points: Sequence[tuple[float, float]],
    template_um: float,
    search_um: float,
    repeats: int,
    *,
    seed: int = 0,
    min_score: float = 0.5,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Estimate template-matching precision by re-matching fixed points under random jitter.

    Each repeat moves the fixed point by up to one fixed pixel, scales the template by 0.8 to 1.2 and moves the
    search start by up to a quarter of the search radius. The moving match is corrected by ``J @ delta`` for the
    fixed-point shift, so the pooled per-axis spread of corrected matches measures matcher precision in the
    moving frame. Matches below ``min_score`` or on the search boundary are rejected.
    """
    from image2image.utils.validation_auto import match_through_transform

    rng = np.random.default_rng(seed)
    step = float(max(fixed_geometry.spacing))
    groups, rejected = [], 0
    for point in points:
        check_cancel(cancel)
        point = np.asarray(point, dtype=float)
        jacobian = transform.jacobian_at(point[None], step)[0]
        matches = []
        for _ in range(repeats):
            delta = rng.uniform(-1, 1, 2) * step
            match = match_through_transform(
                fixed_array,
                fixed_geometry,
                moving_array,
                moving_geometry,
                transform,
                point + delta,
                template_um * rng.uniform(0.8, 1.2),
                search_um,
                initial_offset_um=tuple(rng.uniform(-1, 1, 2) * search_um / 4),
            )
            if not match.ok or match.reason or match.score is None or match.score < min_score:
                rejected += 1
                continue
            matches.append(np.asarray(match.moving_xy) - jacobian @ delta)
        groups.append(np.array(matches).reshape(-1, 2))
    sigma = _pooled_sd(groups)
    return {
        "method": "matcher jitter",
        "sigma_moving_xy": sigma,
        "points": sum(len(group) >= 2 for group in groups),
        "trials": len(points) * repeats,
        "rejected": rejected,
        "reason": "" if sigma is not None else "Too few reliable repeated matches to estimate matcher precision.",
    }


def localization_correction(
    rows: list[dict[str, Any]],
    transform: ValidationTransform,
    manual: dict[str, Any],
    matcher: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Annotate residual rows with expected localization error and summarize the error budget.

    Manual landmarks use the manual fixed and moving precision; assisted landmarks use the manual fixed and the
    matcher moving precision; automatic landmarks use the matcher precision only. Each measured row gains
    ``fle_um`` (expected RMS localization contribution in the moving frame) and ``within_noise`` (squared
    Mahalanobis distance within the 95% chi-square bound, so the residual is explained by clicking alone).

    Returns
    -------
    dict
        ``independent`` (manual and assisted) and ``automatic`` summaries with observed, localization and
        corrected RMS, the localization fraction, within-noise counts and per-axis noise ``sigma_xy``.
    """
    zero = (0.0, 0.0)
    sigmas = {
        "manual": (manual.get("sigma_fixed_xy"), manual.get("sigma_moving_xy")),
        "assisted": (manual.get("sigma_fixed_xy"), matcher.get("sigma_moving_xy")),
        "automatic": (zero, matcher.get("sigma_moving_xy")),
    }
    groups: dict[str, dict[str, list]] = {
        key: {"errors": [], "fle2": [], "diagonal": [], "within": [], "missing": 0}
        for key in ("independent", "automatic")
    }
    for row in rows:
        row["fle_um"], row["within_noise"] = None, None
        if row.get("error_um") is None:
            continue
        group = groups["automatic" if row.get("source") == "automatic" else "independent"]
        fixed_sigma, moving_sigma = sigmas.get(row.get("source", "manual"), sigmas["manual"])
        if fixed_sigma is None or moving_sigma is None:
            group["missing"] += 1
            continue
        jacobian = transform.jacobian_at(np.array([[row["fixed_x"], row["fixed_y"]]]))[0]
        covariance = jacobian @ np.diag(np.square(fixed_sigma)) @ jacobian.T + np.diag(np.square(moving_sigma))
        residual = np.array([row["dx_um"], row["dy_um"]])
        try:
            distance = float(residual @ np.linalg.solve(covariance, residual))
        except np.linalg.LinAlgError:
            distance = np.inf
        row["fle_um"] = float(np.sqrt(np.trace(covariance)))
        row["within_noise"] = bool(distance <= CHI2_2D_95)
        group["errors"].append(row["error_um"])
        group["fle2"].append(float(np.trace(covariance)))
        group["diagonal"].append(np.diag(covariance))
        group["within"].append(row["within_noise"])
    summaries = {}
    for key, group in groups.items():
        count = len(group["errors"])
        summary: dict[str, Any] = {"count": count, "missing": group["missing"]}
        if count:
            observed2 = float(np.mean(np.square(group["errors"])))
            fle2 = float(np.mean(group["fle2"]))
            summary.update(
                observed_rms=float(np.sqrt(observed2)),
                fle_rms=float(np.sqrt(fle2)),
                corrected_rms=float(np.sqrt(max(0.0, observed2 - fle2))),
                fraction_localization=float(min(1.0, fle2 / observed2)) if observed2 > 0 else None,
                within_noise=int(sum(group["within"])),
                sigma_xy=tuple(float(v) for v in np.sqrt(np.mean(group["diagonal"], axis=0))),
            )
        summaries[key] = summary
    return summaries
