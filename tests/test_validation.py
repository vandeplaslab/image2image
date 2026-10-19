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
    ValidationSession,
)
from image2image.utils.validation import (
    AnalysisCancelled,
    ValidationTransform,
    analysis_grid,
    analyze_validation_session,
    deformation_maps,
    discover_validation_candidates,
    distance_statistics,
    export_validation_landmarks,
    import_validation_landmarks,
    inspect_validation_image,
    landmark_errors,
    landmark_exclusions,
    load_validation_session,
    load_validation_transform,
    mask_overlap,
    sample_pixels,
    save_validation_session,
)
from image2image.utils.validation_report import export_validation_report


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


def make_elastix_project(tmp_path: Path, with_preprocessing: bool = False) -> Path:
    """Save a real Elastix project with a known affine resampling chain."""
    from image2image_reg.elastix.transform_sequence import TransformSequence
    from image2image_reg.elastix.transform_utils import affine_to_itk_affine
    from image2image_reg.workflows.elastix import ElastixReg

    session = make_validation_session(tmp_path)
    project = ElastixReg(name="elastix", output_dir=tmp_path)
    for name, image in (("fixed", session.fixed), ("moving", session.moving)):
        project.add_modality(name, image.path, pixel_size=2)
    project.add_registration_path("moving", "fixed", ["affine"])
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
    edge = project.registration_nodes[0]
    edge["registered"] = True
    edge["transforms"] = {"initial": initial, "registration": sequence}
    edge["transform_tag"] = "moving_to_fixed_transformations.json"
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
