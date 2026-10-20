"""Analytic and native-project checks for independent validation."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import SimpleITK as sitk
import tifffile

from image2image.models.validation import (
    ImageGeometry,
    MaskCorrespondence,
    ValidationCandidate,
    ValidationLandmark,
    ValidationPair,
    ValidationSession,
    ValidationStudy,
)
from image2image.utils.validation import (
    AnalysisCancelled,
    ValidationTransform,
    analysis_grid,
    analyze_validation_session,
    analyze_validation_study,
    deformation_maps,
    discover_validation_candidates,
    distance_statistics,
    export_validation_landmarks,
    import_validation_landmarks,
    inspect_validation_image,
    landmark_errors,
    landmark_exclusions,
    load_validation_session,
    load_validation_study,
    load_validation_transform,
    mask_overlap,
    sample_pixels,
    save_validation_session,
    save_validation_study,
    validation_contrast_limits,
    validation_image_pyramid,
)
from image2image.utils.validation_report import export_validation_report, export_validation_study_report


def make_validation_session(tmp_path: Path, spacing: tuple[float, float] = (2, 2)) -> ValidationSession:
    pixels = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40)
    for name, resolution in zip(("fixed", "moving"), spacing, strict=True):
        tifffile.imwrite(
            tmp_path / f"{name}.ome.tiff",
            pixels,
            photometric="minisblack",
            metadata={"axes": "YX", "PhysicalSizeX": resolution, "PhysicalSizeY": resolution},
        )
    matrix = np.eye(3)
    matrix[:2, :2] *= spacing[1] / spacing[0]
    moving = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
    fixed = moving * spacing[1] / spacing[0]
    data = {
        "schema_version": "1.4",
        "tool": "register",
        "transformation_type": "affine",
        "fixed_paths": [{"path": str(tmp_path / "fixed.ome.tiff"), "pixel_size_um": spacing[0]}],
        "moving_paths": [{"path": str(tmp_path / "moving.ome.tiff"), "pixel_size_um": spacing[1]}],
        "fixed_points_yx_px": fixed.tolist(),
        "moving_points_yx_px": moving.tolist(),
        "matrix_yx_px": matrix.tolist(),
        # Deliberately irrelevant physical/display matrices: validation must use native pixel calibration.
        "matrix_xy_um": (np.eye(3) * 10).tolist(),
        "initial_matrix_yx_um": (np.eye(3) * 20).tolist(),
    }
    path = tmp_path / "identity.i2r.json"
    path.write_text(json.dumps(data))
    candidate, f, m = discover_validation_candidates(path)[0]
    return ValidationSession(
        fixed=f,
        moving=m,
        candidates=[candidate],
        landmarks=[
            ValidationLandmark(id="independent", fixed=(20, 24), moving=(23, 28)),
            ValidationLandmark(id="fitting-reuse", fixed=(0, 0), moving=(0, 0)),
        ],
    )


def test_register_calibration_and_legacy(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path, (2, 3))
    candidate = session.candidates[0]
    transform = load_validation_transform(candidate, session)
    np.testing.assert_allclose(transform.map_points([[7, 11], [21, 25]]), [[7, 11], [21, 25]])
    rows, stats = landmark_errors(transform, session.landmarks, landmark_exclusions(session, [transform]))
    assert rows[0]["error_um"] == 5
    assert stats["rms"] == 5 and stats["count"] == 1 and stats["eligible"] == 1
    assert rows[1]["excluded"] == "Reuses a known fitting landmark"
    data = json.loads(Path(candidate.path).read_text())
    del data["matrix_yx_px"]
    Path(candidate.path).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        load_validation_transform(candidate, session)
    candidate.fingerprint = ""
    legacy = load_validation_transform(candidate, session)
    np.testing.assert_allclose(legacy.map_points([[7, 11], [21, 25]]), [[7, 11], [21, 25]], atol=1e-12)


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (np.eye(3), 1),
        (np.array([[1, 0, 3], [0, 1, 4], [0, 0, 1]]), 1),
        (np.array([[0, -1, 80], [1, 0, 0], [0, 0, 1]]), 1),
        (np.diag([0.5, 0.5, 1]), 0.25),
        (np.diag([2, 3, 1]), 6),
        (np.array([[-1, 0, 80], [0, 1, 0], [0, 0, 1]]), -1),
        (np.diag([0, 1, 1]), 0),
    ],
)
def test_analytic_jacobians(matrix: np.ndarray, expected: float) -> None:
    grid = ImageGeometry(shape=(17, 21), spacing=(2, 3), origin=(5, 7))
    candidate = ValidationCandidate(name="synthetic", kind="register", path="unused")
    transform = ValidationTransform(candidate, matrix=matrix)
    maps, summary = deformation_maps(transform, grid, ImageGeometry(shape=(200, 200)))
    assert not maps["valid"][0].any()
    np.testing.assert_allclose(maps["jacobian"][maps["valid"]], expected, atol=1e-6)
    assert summary["nonpositive_fraction"] == (1 if expected <= 0 else 0)
    if expected > 0:
        np.testing.assert_allclose(maps["area_ratio"][maps["valid"]], 1 / expected)
        np.testing.assert_allclose(maps["log2_area_ratio"][maps["valid"]], -np.log2(expected))
    else:
        assert np.isnan(maps["area_ratio"]).all()


def test_spatially_varying_itk_field_and_coverage() -> None:
    grid = ImageGeometry(shape=(25, 31), spacing=(2, 3), origin=(4, 6))
    _yy, xx = np.indices(grid.shape)
    x = grid.origin[0] + xx * grid.spacing[0]
    displacement = np.zeros((*grid.shape, 2))
    displacement[..., 0] = 0.005 * x**2
    image = sitk.GetImageFromArray(displacement, isVector=True)
    image.SetSpacing(grid.spacing)
    image.SetOrigin(grid.origin)
    transform = ValidationTransform(
        ValidationCandidate(name="field", kind="elastix", path="unused"),
        itk_transform=sitk.DisplacementFieldTransform(image),
    )
    maps, _ = deformation_maps(transform, grid, ImageGeometry(shape=(100, 100)))
    np.testing.assert_allclose(maps["jacobian"][1:-1, 1:-1], (1 + 0.01 * x)[1:-1, 1:-1], rtol=1e-6)
    _, summary = deformation_maps(transform, grid, ImageGeometry(shape=(10, 10)))
    assert summary["coverage"] < 1


def test_common_exclusions_and_missing_evidence(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    a = load_validation_transform(session.candidates[0], session)
    b = ValidationTransform(a.candidate, matrix=np.eye(3), fitting=[session.landmarks[0]])
    exclusions = landmark_exclusions(session, [a, b])
    assert set(exclusions) == {point.id for point in session.landmarks}
    for transform in (a, b):
        _, metrics = landmark_errors(transform, session.landmarks, exclusions)
        assert metrics["count"] == metrics["eligible"] == 0
        assert metrics["rms"] is None
    assert distance_statistics(np.array([]))["mean"] is None


@pytest.mark.parametrize(
    "fixed,moving,dice,iou",
    [
        ([0, 0], [0, 0], None, None),
        ([1, 0], [0, 0], 0, 0),
        ([1, 0], [1, 0], 1, 1),
        ([1, 1], [1, 0], 2 / 3, 0.5),
    ],
)
def test_mask_scores(fixed: list, moving: list, dice: float | None, iou: float | None) -> None:
    result = mask_overlap(np.array(fixed, bool), np.array(moving, bool), 6)
    assert result["dice"] == dice
    assert result["iou"] == iou
    assert result["fixed_area_um2"] == sum(fixed) * 6


def test_bounded_nearest_neighbor_sampling() -> None:
    class TrackedArray:
        dtype = np.dtype("uint16")

        def __getitem__(self, item: tuple[slice, slice]) -> np.ndarray:
            y, x = item
            shape = (y.stop - y.start, x.stop - x.start)
            assert np.prod(shape) * 2 <= 128
            yy, xx = np.indices(shape)
            return ((xx + x.start + yy + y.start) % 3).astype(self.dtype)

    geometry = ImageGeometry(shape=(100, 100), spacing=(2, 3), origin=(10, 20))
    coords = geometry.to_physical(np.array([[1.1, 2.1], [98.8, 98.8], [300, 300]]))
    sampled, coverage = sample_pixels(TrackedArray(), geometry, coords, labels=True, max_source_bytes=128)
    np.testing.assert_array_equal(sampled, [0, 0, 0])
    np.testing.assert_array_equal(coverage, [True, True, False])
    with pytest.raises(ValueError, match="integer"):
        sample_pixels(np.ones((4, 4), float), ImageGeometry(shape=(4, 4)), np.array([[1, 1]]), labels=True)


def test_csv_and_session_roundtrip(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path, (2, 3))
    path = tmp_path / "points.csv"
    for units in ("px", "um"):
        session.landmark_units = units
        export_validation_landmarks(path, session)
        assert import_validation_landmarks(path, session, units) == session.landmarks
    for text in (
        "id,x,y\na,1,2\n",
        "id,fixed_x,fixed_y,moving_x,moving_y\na,nan,2,3,4\n",
        "id,fixed_x,fixed_y,moving_x,moving_y\na,1,2,3,4\na,4,5,6,7\n",
        "id,fixed_x,fixed_y,moving_x,moving_y\na,1,2,3,4,5\n",
        "id,fixed_x,fixed_y,moving_x,moving_y,id\na,1,2,3,4,b\n",
    ):
        path.write_text(text)
        with pytest.raises(ValueError):
            import_validation_landmarks(path, session, "px")
    saved = save_validation_session(tmp_path / "session", session)
    loaded, changes = load_validation_session(saved)
    assert loaded == session and changes == []
    Path(session.candidates[0].path).write_text("{}")
    _, changes = load_validation_session(saved)
    assert "Registration changed" in changes[0]


def test_grid_cap_and_cancel(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    session.fixed.geometry = ImageGeometry(shape=(80001, 100001), spacing=(0.5, 1))
    grid = analysis_grid(session)
    assert np.prod(grid.shape) <= 4_000_000
    assert grid.spacing[1] == 2 * grid.spacing[0]
    cancel = Event()
    cancel.set()
    with pytest.raises(AnalysisCancelled):
        analyze_validation_session(session, tmp_path / "maps", cancel)


def test_full_analysis_and_report(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    for which, label in (("fixed", 3), ("moving", 9)):
        mask = np.zeros((32, 40), np.uint8)
        mask[4:20, 8:30] = label
        path = tmp_path / f"{which}-mask.tif"
        tifffile.imwrite(path, mask)
        setattr(session, f"{which}_mask", inspect_validation_image(path, (2, 2)))
    session.mask_pairs = [MaskCorrespondence(name="tissue", fixed_label=3, moving_label=9)]
    result = analyze_validation_session(session, tmp_path / "maps")
    candidate = result.candidates[0]
    assert candidate.validation["rms"] == 5
    assert candidate.fitting["rms"] == 0
    assert candidate.masks[0]["dice"] == 1
    assert candidate.deformation["nonpositive_fraction"] == 0
    with np.load(candidate.map_path) as maps:
        np.testing.assert_array_equal(maps["fixed_image"], maps["registered_image"])
        assert set(np.unique(maps["moving_mask"])) == {0, 9}
    path = export_validation_report(result, tmp_path / "report")
    text = path.read_text()
    assert "data:image/png;base64," in text
    assert "G(fixed) - moving" in text
    assert "Reuses a known fitting landmark" in text
    assert "http://" not in text and "https://" not in text
    data = json.loads((path.parent / "results.json").read_text())
    assert data["candidates"][0]["validation"]["rms"] == 5
    assert data["session"]["fixed"]["fingerprint"]
    assert (path.parent / "candidate-1.npz").exists()
    with pytest.raises(ValueError, match="empty"):
        export_validation_report(result, path.parent)


def make_elastix_project(tmp_path: Path, with_preprocessing: bool = False, *, multiple: bool = False) -> Path:
    """Save a real Elastix project with a known affine resampling chain."""
    from image2image_reg.elastix.transform_sequence import TransformSequence
    from image2image_reg.elastix.transform_utils import affine_to_itk_affine
    from image2image_reg.workflows.elastix import ElastixReg

    session = make_validation_session(tmp_path)
    project = ElastixReg(name="elastix", output_dir=tmp_path)
    for name, image in (("fixed", session.fixed), ("moving", session.moving)):
        project.add_modality(name, image.path, pixel_size=2)
    project.add_registration_path("moving", "fixed", ["affine"])
    if multiple:
        second = tmp_path / "moving-second.ome.tiff"
        second.write_bytes(Path(session.moving.path).read_bytes())
        project.add_modality("moving-second", second, pixel_size=2)
        project.add_registration_path("moving-second", "fixed", ["affine"])
    project.save()
    matrix = np.array([[1, 0, 2], [0, 1, 4], [0, 0, 1]])  # YX physical pull transform
    parameters = affine_to_itk_affine(matrix, (40, 32), spacing=2)
    sequence = TransformSequence.from_list([parameters])
    initial = None
    if with_preprocessing:
        initial = TransformSequence.from_list(
            [affine_to_itk_affine(np.array([[2, 0, 3], [0, 3, 5], [0, 0, 1]]), (40, 32), spacing=2)]
        )
        project.original_size_transforms["fixed"] = [
            affine_to_itk_affine(np.array([[1, 0, -6], [0, 1, -8], [0, 0, 1]]), (40, 32), spacing=2)
        ]
    for edge in project.registration_nodes:
        edge["registered"] = True
        edge["transforms"] = {"initial": initial, "registration": sequence}
        edge["transform_tag"] = f"{edge['modalities']['source']}_to_fixed_transformations.json"
    project.transformations = project._collate_transformations()
    project.save_transformations()
    project.save(registered=True)
    return project.project_dir


def test_real_elastix_loading(tmp_path: Path) -> None:
    path = make_elastix_project(tmp_path)
    c, fixed, moving = discover_validation_candidates(path)[0]
    session = ValidationSession(fixed=fixed, moving=moving, candidates=[c])
    transform = load_validation_transform(c, session)
    np.testing.assert_allclose(transform.map_points([[10, 12]]), [[14, 14]])
    result = analyze_validation_session(session, tmp_path / "maps")
    assert result.candidates[0].validation["rms"] is None
    assert result.candidates[0].deformation["median"] == pytest.approx(1)


def test_pair_mismatch_and_invalid_matrix(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    session.fixed.geometry.spacing = (4, 4)
    with pytest.raises(ValueError, match="calibration"):
        load_validation_transform(session.candidates[0], session)
    session.fixed.geometry.spacing = (2, 2)
    candidate = session.candidates[0]
    data = json.loads(Path(candidate.path).read_text())
    data["matrix_yx_px"] = np.zeros((3, 3)).tolist()
    Path(candidate.path).write_text(json.dumps(data))
    candidate.fingerprint = ""
    with pytest.raises(ValueError, match="singular"):
        load_validation_transform(candidate, session)
    data["fixed_paths"][0]["image_shape"] = [10, 10]
    Path(candidate.path).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="saved Register image shape"):
        discover_validation_candidates(candidate.path)
    with pytest.raises(FileNotFoundError, match="Image missing"):
        inspect_validation_image(tmp_path / "missing.tif")


def test_elastix_original_reference_and_initial_composition(tmp_path: Path) -> None:
    path = make_elastix_project(tmp_path, with_preprocessing=True)
    candidate, fixed, moving = discover_validation_candidates(path)[0]
    session = ValidationSession(fixed=fixed, moving=moving, candidates=[candidate])
    transform = load_validation_transform(candidate, session)
    # Original-reference shift, fitted translation, then the source initial scale and translation.
    np.testing.assert_allclose(transform.map_points([[10, 12]]), [[23, 19]])


def test_legacy_image_references_and_report_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_validation_session(tmp_path)
    path = Path(session.candidates[0].path)
    data = json.loads(path.read_text())
    data["schema_version"] = "1.1"
    data["fixed_paths"] = [data["fixed_paths"][0]["path"]]
    data["moving_paths"] = [data["moving_paths"][0]["path"]]
    del data["matrix_yx_px"]
    path.write_text(json.dumps(data))
    candidate, _fixed, _moving = discover_validation_candidates(path)[0]
    session.candidates = [candidate]
    np.testing.assert_allclose(load_validation_transform(candidate, session).map_points([[12, 14]]), [[12, 14]])
    result = analyze_validation_session(session, tmp_path / "maps")

    def fail(*_args, **_kwargs) -> bytes:
        raise RuntimeError("Rendering failed")

    monkeypatch.setattr("image2image.utils.validation_report.plot_validation_candidate", fail)
    with pytest.raises(RuntimeError, match="Rendering failed"):
        export_validation_report(result, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_changed_input_during_analysis(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)

    def change_input(message: str) -> None:
        if message.startswith("Analyzing"):
            Path(session.candidates[0].path).write_text("{}")

    with pytest.raises(ValueError, match="changed during"):
        analyze_validation_session(session, tmp_path / "maps", progress=change_input)


def test_real_elastix_bspline_chain(tmp_path: Path) -> None:
    from copy import deepcopy

    from image2image_reg.elastix.transformation_map import BASE_AFFINE_TRANSFORM

    path = make_elastix_project(tmp_path)
    reference = sitk.Image([40, 32], sitk.sitkFloat32)
    reference.SetSpacing((2, 2))
    spline = sitk.BSplineTransformInitializer(reference, [4, 4])
    parameters = np.asarray(spline.GetParameters())
    parameters[: len(parameters) // 2] = np.linspace(-2, 2, len(parameters) // 2) ** 2
    spline.SetParameters(parameters)
    fixed_parameters = spline.GetFixedParameters()
    data = deepcopy(BASE_AFFINE_TRANSFORM)
    data.update(
        Transform=["BSplineTransform"],
        Size=["40", "32"],
        Spacing=["2", "2"],
        Origin=["0", "0"],
        Direction=["1", "0", "0", "1"],
        GridSize=[str(int(v)) for v in fixed_parameters[:2]],
        GridOrigin=[str(v) for v in fixed_parameters[2:4]],
        GridSpacing=[str(v) for v in fixed_parameters[4:6]],
        GridDirection=[str(v) for v in fixed_parameters[6:]],
        TransformParameters=[str(v) for v in parameters],
    )
    transforms_path = path / "Transformations" / "moving_to_fixed_transformations.json"
    payload = json.loads(transforms_path.read_text())
    payload["000-to-fixed"].append(data)
    transforms_path.write_text(json.dumps(payload))
    candidate, fixed, moving = discover_validation_candidates(path)[0]
    session = ValidationSession(fixed=fixed, moving=moving, candidates=[candidate])
    loaded = load_validation_transform(candidate, session)
    points = np.array([[20, 20], [30, 40], [60, 20]], dtype=float)
    expected = np.array([spline.TransformPoint(tuple(point)) for point in points]) + [4, 2]
    np.testing.assert_allclose(loaded.map_points(points), expected, atol=1e-10)
    maps, _ = deformation_maps(loaded, analysis_grid(session), moving.geometry)
    assert np.ptp(maps["jacobian"][maps["valid"]]) > 0


def test_two_candidates_use_same_evidence(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    data = json.loads(Path(session.candidates[0].path).read_text())
    matrix = np.eye(3)
    matrix[:2, 2] = [-2, -1.5]  # G shifts X by 3 um, Y by 4 um.
    data["matrix_yx_px"] = matrix.tolist()
    path = tmp_path / "second.i2r.json"
    path.write_text(json.dumps(data))
    session.candidates.append(discover_validation_candidates(path)[0][0])
    result = analyze_validation_session(session, tmp_path / "maps")
    assert [c.validation["count"] for c in result.candidates] == [1, 1]
    assert [c.validation["rms"] for c in result.candidates] == [5, 0]
    assert [c.fitting["rms"] for c in result.candidates] == [0, 5]


def test_elastix_through_chain_and_missing_stage(tmp_path: Path) -> None:
    from image2image_reg.elastix.transform_sequence import TransformSequence
    from image2image_reg.elastix.transform_utils import affine_to_itk_affine
    from image2image_reg.workflows.elastix import ElastixReg

    session = make_validation_session(tmp_path)
    project = ElastixReg("through", tmp_path)
    for name, image in (("fixed", session.fixed), ("middle", session.fixed), ("moving", session.moving)):
        project.add_modality(name, image.path, pixel_size=2)
    project.add_registration_path("middle", "fixed", ["affine"])
    project.add_registration_path("moving", "fixed", ["affine"], through="middle")
    project.save()
    for edge in project.registration_nodes:
        edge["registered"] = True
        edge["transforms"] = {
            "initial": None,
            "registration": TransformSequence.from_list(
                [affine_to_itk_affine(np.array([[1, 0, 2], [0, 1, 4], [0, 0, 1]]), (40, 32), spacing=2)]
            ),
        }
    project.transformations = project._collate_transformations()
    project.save_transformations()
    project.save(registered=True)
    candidate, fixed, moving = next(
        item for item in discover_validation_candidates(project.project_dir) if item[0].source == "moving"
    )
    session = ValidationSession(fixed=fixed, moving=moving, candidates=[candidate])
    np.testing.assert_allclose(load_validation_transform(candidate, session).map_points([[10, 12]]), [[18, 16]])
    path = project.transformations_dir / "moving_to_fixed_transformations.json"
    data = json.loads(path.read_text())
    del data["001-to-fixed"]
    path.write_text(json.dumps(data))
    candidate.fingerprint = ""
    with pytest.raises(ValueError, match="missing a stage"):
        load_validation_transform(candidate, session)


def make_validation_study(tmp_path: Path) -> ValidationStudy:
    first = make_validation_session(tmp_path)
    second_path = tmp_path / "moving-second.ome.tiff"
    second_path.write_bytes(Path(first.moving.path).read_bytes())
    config = json.loads(Path(first.candidates[0].path).read_text())
    config["moving_paths"][0]["path"] = str(second_path)
    registration = tmp_path / "second.i2r.json"
    registration.write_text(json.dumps(config))
    candidate, fixed, moving = discover_validation_candidates(registration)[0]
    second = first.model_copy(deep=True)
    second.fixed, second.moving, second.candidates = fixed, moving, [candidate]
    second.landmarks[0].moving = (26, 32)
    return ValidationStudy(pairs=[ValidationPair(session=first), ValidationPair(session=second)])


def test_study_roundtrip_legacy_and_shared_reference(tmp_path: Path) -> None:
    study = make_validation_study(tmp_path)
    path = save_validation_study(tmp_path / "study", study)
    loaded, changes = load_validation_study(path)
    assert loaded == study and not changes
    legacy_path = save_validation_session(tmp_path / "legacy", study.pairs[0].session)
    legacy, changes = load_validation_study(legacy_path)
    assert len(legacy.pairs) == 1 and legacy.pairs[0].session == study.pairs[0].session
    assert not changes
    invalid = study.model_copy(deep=True)
    invalid.pairs[1].session.fixed.geometry.spacing = (3, 3)
    with pytest.raises(ValueError, match="same fixed image"):
        save_validation_study(path, invalid)
    assert load_validation_study(path)[0] == study
    payload = study.model_dump()
    payload["pairs"][1]["id"] = payload["pairs"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        ValidationStudy.model_validate(payload)


def test_study_analysis_failure_isolation_and_export(tmp_path: Path) -> None:
    study = make_validation_study(tmp_path)
    result = analyze_validation_study(study, tmp_path / "maps")
    assert [pair.analysis.candidates[0].validation["rms"] for pair in result.pairs] == [5, 10]
    assert result.pairs[0].analysis.grid == result.pairs[1].analysis.grid
    Path(study.pairs[0].session.candidates[0].path).unlink()
    partial = analyze_validation_study(study, tmp_path / "partial")
    assert partial.pairs[0].analysis is None and "Registration missing" in partial.pairs[0].error
    assert partial.pairs[1].analysis.candidates[0].validation["rms"] == 10
    report = export_validation_study_report(partial, tmp_path / "study-report")
    text = report.read_text()
    assert "Analysis failed" in text and "data:image/png;base64," in text
    import csv

    with (report.parent / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["pair_id"] for row in rows] == [pair.id for pair in study.pairs]
    assert rows[0]["status"] == "failed" and float(rows[1]["validation_rms_um"]) == 10
    payload = json.loads((report.parent / "results.json").read_text())
    for record in payload["pairs"]:
        if record["analysis"]:
            for candidate in record["analysis"]["candidates"]:
                assert (report.parent / candidate["map_path"]).exists()
    cancel = Event()
    cancel.set()
    with pytest.raises(AnalysisCancelled):
        analyze_validation_study(study, tmp_path / "cancelled", cancel)


def test_incomplete_masks_do_not_block_other_measurements(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    session.fixed_mask = session.fixed.model_copy(deep=True)
    result = analyze_validation_session(session, tmp_path / "maps")
    assert result.candidates[0].validation["rms"] == 5
    assert result.candidates[0].fitting["count"] == 3
    assert result.candidates[0].masks == []
    assert result.candidates[0].deformation["coverage"] == 1


def test_display_pyramids_are_lazy_and_split_rgb(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import dask.array as da
    from dask import delayed

    from image2image.utils.validation import open_validation_reader

    reads = []

    @delayed
    def source() -> np.ndarray:
        reads.append(True)
        return np.ones((2048, 4096), dtype=np.uint16)

    array = da.from_delayed(source(), shape=(2048, 4096), dtype=np.uint16)
    reader = SimpleNamespace(n_in_pyramid=1, get_channel=lambda index, level, split_rgb: array)
    levels = validation_image_pyramid(reader, 0)
    assert reads == [] and levels[0] is array
    assert [level.shape for level in levels] == [(2048, 4096), (1024, 2048), (512, 1024)]
    np.testing.assert_array_equal(levels[-1].compute(), 1)
    assert reads == [True]
    path = tmp_path / "rgb.ome.tiff"
    pixels = np.zeros((32, 40, 3), dtype=np.uint8)
    pixels[..., 1] = 77
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(pixels, subifds=1, photometric="rgb", metadata={"axes": "YXS"})
        writer.write(pixels[::2, ::2], subfiletype=1, photometric="rgb")
    image = inspect_validation_image(path, spacing=(2, 3))
    native = open_validation_reader(image)
    try:
        levels = validation_image_pyramid(native, 1)
        assert [level.shape for level in levels] == [(32, 40), (16, 20)]
        np.testing.assert_array_equal(levels[0].compute(), 77)
    finally:
        native.close()


def test_calibration_provenance(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    assert session.fixed.calibration_source == "registration" and session.fixed.calibration_confirmed
    unconfirmed = inspect_validation_image(session.fixed.path)
    assert unconfirmed.calibration_source == "unconfirmed" and not unconfirmed.calibration_confirmed
    explicit = inspect_validation_image(session.fixed.path, spacing=(1, 1))
    assert explicit.calibration_source == "user" and explicit.calibration_confirmed


def test_anisotropic_tiff_metadata_and_manual_legacy_calibration(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    ome = tmp_path / "anisotropic.ome.tiff"
    tifffile.imwrite(
        ome,
        np.ones((8, 10), dtype=np.uint8),
        metadata={
            "axes": "YX",
            "PhysicalSizeX": 500,
            "PhysicalSizeXUnit": "nm",
            "PhysicalSizeY": 3000,
            "PhysicalSizeYUnit": "nm",
        },
    )
    assert inspect_validation_image(ome).geometry.spacing == (0.5, 3)
    tags = tmp_path / "anisotropic.tiff"
    tifffile.imwrite(tags, np.ones((8, 10), dtype=np.uint8), resolution=(20000, 4000), resolutionunit="CENTIMETER")
    assert inspect_validation_image(tags).geometry.spacing == (0.5, 2.5)
    registration = Path(session.candidates[0].path)
    config = json.loads(registration.read_text())
    for key in ("fixed_paths", "moving_paths"):
        del config[key][0]["pixel_size_um"]
    registration.write_text(json.dumps(config))
    candidate, fixed, moving = discover_validation_candidates(registration)[0]
    for image in (fixed, moving):
        image.geometry.spacing, image.geometry.origin = (3, 5), (7, 11)
        image.calibration_source, image.calibration_confirmed = "user", True
    custom = ValidationSession(fixed=fixed, moving=moving, candidates=[candidate])
    transform = load_validation_transform(candidate, custom)
    np.testing.assert_allclose(transform.map_points([[19, 31]]), [[19, 31]])
    assert transform.fitting[0].fixed == (7, 11)


def test_contrast_limits_use_coarsest_level_and_preserve_small_image_range() -> None:
    small = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40)
    low, high = validation_contrast_limits([small])
    assert 0 < low < 20 and 1200 < high < 1280
    assert validation_contrast_limits([np.ones((2, 2), dtype=np.uint8) * 77]) == (77, 78)
    assert validation_contrast_limits([np.full((2, 2), np.nan)]) == (0, 1)

    class Coarsest:
        shape = (8192, 8192)

        def __getitem__(self, slices: tuple[slice, slice]) -> np.ndarray:
            assert slices[0].step == slices[1].step == 32
            return small

    assert validation_contrast_limits([object(), Coarsest()]) == (low, high)


def test_aligned_images_without_registration_project(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    Path(session.candidates[0].path).unlink()
    session.candidates = []
    session.landmarks = session.landmarks[:1]
    session.fixed_mask, session.moving_mask = session.fixed.model_copy(), session.moving.model_copy()
    session.mask_pairs = [MaskCorrespondence(name="region", fixed_label=1, moving_label=1)]
    result = analyze_validation_session(session, tmp_path / "direct-maps")
    candidate = result.candidates[0]
    assert candidate.name == "Already aligned images"
    assert candidate.validation["count"] == 1 and candidate.validation["rms"] == 5
    assert candidate.fitting["rms"] is None and "not fitted" in candidate.fitting["reason"]
    assert candidate.masks[0]["dice"] == 1 and candidate.masks[0]["iou"] == 1
    assert (
        candidate.deformation["median"] == 1
        and "original registration deformation is unknown" in candidate.deformation["reference"]
    )
    report = export_validation_report(result, tmp_path / "direct-report")
    assert "original registration deformation is unknown" in report.read_text()
    reopened, changes = load_validation_session(report.parent / "session.i2validate.json")
    assert not changes
    assert analyze_validation_session(reopened, tmp_path / "reopened-maps").candidates[0].validation["rms"] == 5


def test_register_user_calibration_preserves_pixel_transform(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    session.fixed.geometry.spacing, session.fixed.geometry.origin = (3, 5), (7, 11)
    session.moving.geometry.spacing, session.moving.geometry.origin = (6, 10), (13, 17)
    session.fixed.calibration_source = session.moving.calibration_source = "user"
    transform = load_validation_transform(session.candidates[0], session)
    fixed_pixels = np.array([[2, 4], [6, 8]])
    np.testing.assert_allclose(
        transform.map_points(session.fixed.geometry.to_physical(fixed_pixels)),
        session.moving.geometry.to_physical(fixed_pixels),
    )
    assert transform.fitting[0].fixed == (7, 11) and transform.fitting[0].moving == (13, 17)


def test_elastix_user_calibration_reexpresses_saved_physical_transform(tmp_path: Path) -> None:
    path = make_elastix_project(tmp_path)
    candidate, fixed, moving = discover_validation_candidates(path)[0]
    original = ValidationSession(fixed=fixed, moving=moving, candidates=[candidate])
    base = load_validation_transform(candidate, original)
    edited = original.model_copy(deep=True)
    edited.fixed.geometry.spacing, edited.fixed.geometry.origin = (3, 5), (7, 11)
    edited.moving.geometry.spacing, edited.moving.geometry.origin = (6, 10), (13, 17)
    edited.fixed.calibration_source = edited.moving.calibration_source = "user"
    updated = load_validation_transform(candidate, edited)
    pixels = np.array([[5, 6], [10, 12]])
    expected_pixels = moving.geometry.to_pixels(base.map_points(fixed.geometry.to_physical(pixels)))
    expected = edited.moving.geometry.to_physical(expected_pixels)
    np.testing.assert_allclose(updated.map_points(edited.fixed.geometry.to_physical(pixels)), expected)
    grid = ImageGeometry(shape=(5, 6), spacing=(3, 5), origin=(7, 11))
    xy = grid.to_physical(np.stack(np.indices(grid.shape)[::-1], axis=-1))
    np.testing.assert_allclose(updated.map_grid(grid), updated.map_points(xy.reshape(-1, 2)).reshape(5, 6, 2))


def test_rgb_display_levels_preserve_colors_and_lazy_spatial_reduction(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import dask.array as da
    from dask import delayed

    reads = []

    @delayed
    def source() -> np.ndarray:
        reads.append(True)
        return np.broadcast_to(np.array([11, 77, 203], dtype=np.uint8), (2048, 3072, 3)).copy()

    array = da.from_delayed(source(), shape=(2048, 3072, 3), dtype=np.uint8)
    reader = SimpleNamespace(n_in_pyramid=1, get_channel=lambda index, level, split_rgb: array)
    levels = validation_image_pyramid(reader, 0, rgb=True)
    assert not reads and levels[0] is array
    assert [level.shape for level in levels] == [(2048, 3072, 3), (1024, 1536, 3), (512, 768, 3)]
    np.testing.assert_array_equal(levels[-1][0, 0].compute(), [11, 77, 203])
    assert reads == [True]


def test_display_selection_roundtrip_and_legacy_default(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    session.fixed.display_channel = "rgb"
    session.moving.display_channel = 1
    study, _ = load_validation_study(save_validation_session(tmp_path / "display", session))
    assert study.pairs[0].session.fixed.display_channel == "rgb"
    assert study.pairs[0].session.moving.display_channel == 1
    assert study.pairs[0].session.moving.channel == 0
    payload = session.model_dump()
    for image in (payload["fixed"], payload["moving"]):
        del image["display_channel"], image["is_rgb"]
    legacy = ValidationSession.model_validate(payload)
    assert legacy.fixed.display_channel is None and legacy.fixed.channel == 0
    with pytest.raises(ValueError, match="nonnegative"):
        legacy.fixed.display_channel = -1


def test_elastix_discovery_collects_failed_paths_without_hiding_valid_pairs(tmp_path: Path) -> None:
    path = make_elastix_project(tmp_path, multiple=True)
    config_path = path / "project.config.json"
    config = json.loads(config_path.read_text())
    config["registration_paths"]["bad"] = {"source": "missing", "target": "fixed"}
    config_path.write_text(json.dumps(config))
    errors = []
    choices = discover_validation_candidates(path, errors=errors)
    assert {candidate.source for candidate, _, _ in choices} == {"moving", "moving-second"}
    assert len(errors) == 1 and "missing to fixed" in errors[0]
    with pytest.raises(KeyError, match="missing"):
        discover_validation_candidates(path)


def test_elastix_candidate_loading_isolated_from_other_missing_sources(tmp_path: Path) -> None:
    path = make_elastix_project(tmp_path, multiple=True)
    (tmp_path / "moving-second.ome.tiff").unlink()
    errors = []
    choices = discover_validation_candidates(path, errors=errors)
    assert len(choices) == len(errors) == 1
    candidate, fixed, moving = choices[0]
    transform = load_validation_transform(candidate, ValidationSession(fixed=fixed, moving=moving))
    np.testing.assert_allclose(transform.map_points([[10, 12]]), [[14, 14]])


def test_elastix_missing_candidate_transform_fails_only_its_pair(tmp_path: Path) -> None:
    path = make_elastix_project(tmp_path, multiple=True)
    (path / "Transformations" / "moving-second_to_fixed_transformations.json").unlink()
    study = ValidationStudy(
        pairs=[
            ValidationPair(session=ValidationSession(fixed=fixed, moving=moving, candidates=[candidate]))
            for candidate, fixed, moving in discover_validation_candidates(path)
        ]
    )
    result = analyze_validation_study(study, tmp_path / "isolated-results")
    assert result.pairs[0].analysis is not None and not result.pairs[0].error
    assert result.pairs[1].analysis is None and "incomplete" in result.pairs[1].error
