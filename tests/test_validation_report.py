"""Report layout, PDF generation, privacy and atomic export checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from image2image.models.validation import PairAnalysis, StudyAnalysis, ValidationPair, ValidationStudy
from image2image.utils.validation import analyze_validation_session
from image2image.utils.validation_report import (
    _redact_paths,
    export_validation_report,
    export_validation_study_report,
)
from test_validation import make_validation_session


@pytest.mark.parametrize("include_paths", [False, True])
def test_report_layout_pdf_and_privacy(tmp_path: Path, include_paths: bool) -> None:
    source = tmp_path / "private source directory"
    source.mkdir()
    analysis = analyze_validation_session(make_validation_session(source), tmp_path / "maps")
    fixed_path = analysis.session.fixed.path
    analysis.session.independence_note = f"Reviewed {fixed_path}\nKeep <review> notes."
    analysis.session.fixed.reader_kws = {"nested": {"reference": r"C:\private folder\calibration file.json"}}
    analysis.candidates[0].name = f"Registration: {analysis.session.candidates[0].path}"
    analysis.candidates[0].landmarks[1]["excluded"] += f"; see '{fixed_path}'"
    original = analysis.session.model_dump(mode="json")
    original_name = analysis.candidates[0].name
    report = export_validation_report(analysis, tmp_path / "report", include_paths=include_paths)
    text = report.read_text()
    assert "<h2>Comparison summary</h2>" in text
    assert "<summary>Methods and interpretation</summary>" in text
    assert "<summary>Inputs, calibration and analysis settings</summary>" in text
    assert "<strong>Validation independence:</strong>" in text
    assert "&lt;review&gt;" in text and "<review>" not in text
    assert "<th scope='col'>RMS (um)</th>" in text
    assert 'href="report.pdf"' in text
    assert "Dice = 2 * intersection" in text and "central differences" in text
    pdf = (report.parent / "report.pdf").read_bytes()
    assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
    payload = json.loads((report.parent / "results.json").read_text())
    assert payload["candidates"][0]["validation"]["rms"] == 5
    assert payload["candidates"][0]["map_path"] == "candidate-1.npz"
    assert payload["session"]["fixed"]["path"] == (fixed_path if include_paths else Path(fixed_path).name)
    for path in report.parent.iterdir():
        if not include_paths and path.suffix in (".json", ".csv", ".html"):
            assert str(source) not in path.read_text()
            assert "private folder" not in path.read_text()
    assert analysis.session.model_dump(mode="json") == original
    assert analysis.candidates[0].name == original_name
    with (
        np.load(analysis.candidates[0].map_path) as original_maps,
        np.load(report.parent / "candidate-1.npz") as exported,
    ):
        for key in original_maps.files:
            np.testing.assert_array_equal(exported[key], original_maps[key])


def test_study_report_privacy_and_pair_statuses(tmp_path: Path) -> None:
    session = make_validation_session(tmp_path)
    analysis = analyze_validation_session(session, tmp_path / "maps")
    pairs = [ValidationPair(session=session.model_copy(deep=True)) for _ in range(3)]
    study = ValidationStudy(pairs=pairs)
    error = f"Cannot read '{session.moving.path}' or C:\\private\\missing.tif; preview {tmp_path / 'missing file.tif'}"
    outcomes = [
        PairAnalysis(pairs[0].id, pairs[0].name, analysis),
        PairAnalysis(pairs[1].id, pairs[1].name, error=error),
        PairAnalysis(pairs[2].id, pairs[2].name),
    ]
    report = export_validation_study_report(StudyAnalysis(study, outcomes), tmp_path / "report", include_paths=False)
    payload = json.loads((report.parent / "results.json").read_text())
    assert [pair["status"] for pair in payload["pairs"]] == ["complete", "failed", "not_analyzed"]
    assert payload["pairs"][1]["error"] == "Cannot read 'moving.ome.tiff' or missing.tif; preview missing file.tif"
    assert payload["pairs"][0]["analysis"]["candidates"][0]["map_path"] == "pair-1/candidate-1.npz"
    assert "Analysis failed" in report.read_text() and "Not analyzed" in report.read_text()
    for path in report.parent.rglob("*"):
        if path.suffix in (".json", ".csv", ".html"):
            assert str(tmp_path) not in path.read_text()
            assert "C:\\private" not in path.read_text()
    assert (report.parent / "report.pdf").read_bytes().startswith(b"%PDF-")
    assert (report.parent / "pair-1" / "report.pdf").read_bytes().startswith(b"%PDF-")
    assert not (report.parent / "pair-2").exists()
    assert study.pairs[0].session.fixed.path == session.fixed.path and outcomes[1].error == error


@pytest.mark.parametrize(
    "value, expected",
    [
        ("/private folder/study/image.tif", "image.tif"),
        (r"C:\private folder\study\image.tif", "image.tif"),
        (r"\\server\private folder\image.tif", "image.tif"),
        ("Missing '/private folder/study/image file.tif'", "Missing 'image file.tif'"),
        ("Missing /private/study/image.tif", "Missing image.tif"),
        ("Missing relative/study/image.tif", "Missing image.tif"),
        ("Used/eligible: 1/2; RMS 5 um", "Used/eligible: 1/2; RMS 5 um"),
    ],
)
def test_redact_paths(value: str, expected: str) -> None:
    assert _redact_paths({"note": value}) == {"note": expected}


def test_pdf_failure_keeps_existing_empty_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = analyze_validation_session(make_validation_session(tmp_path), tmp_path / "maps")
    output = tmp_path / "report"
    output.mkdir()

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("PDF unavailable")

    monkeypatch.setattr("image2image.utils.validation_report._write_pdf_report", fail)
    with pytest.raises(OSError, match="PDF unavailable"):
        export_validation_report(analysis, output)
    assert output.is_dir() and not list(output.iterdir())
    assert not list(tmp_path.glob(".i2validate-report-*"))
