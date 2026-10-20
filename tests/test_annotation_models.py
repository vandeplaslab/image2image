"""Model catalogues, atomic managed downloads, and offline cache reuse."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from image2image.utils.annotation import model_cache
from image2image.utils.annotation.worker import run_job


@pytest.fixture
def catalogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    payloads = {"encoder": b"weights" * 90000, "encoder_decoder": b"decoder" * 3000}
    entries = [
        {
            "name": "encoder",
            "reason": "",
            "assets": [
                {
                    "name": name,
                    "url": f"https://models.example/{name}",
                    "hash": "sha256:" + hashlib.sha256(data).hexdigest(),
                }
                for name, data in payloads.items()
            ],
        }
    ]
    monkeypatch.setattr(model_cache, "USER_CACHE_DIR", tmp_path)
    monkeypatch.setattr(model_cache, "_registry", lambda _backend: entries)
    return entries, payloads


class Response(io.BytesIO):
    """In-memory HTTP body with the same headers consumed by the downloader."""

    def __init__(self, data: bytes, *, size: int | None = None) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if size is None else size)}


def test_download_cache_reuse_and_no_discovery_network(catalogue: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    _entries, payloads = catalogue
    calls = []

    def fetch(request: Any, **_kwargs: Any) -> Response:
        name = request.full_url.rsplit("/", 1)[1]
        calls.append(name)
        return Response(payloads[name])

    monkeypatch.setattr(model_cache, "urlopen", fetch)
    assert not model_cache.list_pretrained_models("micro-sam")[0]["cached"]
    assert calls == []
    progress = []
    result = model_cache.download_pretrained_model(
        "micro-sam", "encoder", task_progress=lambda *args: progress.append(args)
    )
    assert result["cached"]
    assert calls == ["encoder", "encoder_decoder"]
    assert progress[-1][1:3] == (len(payloads["encoder_decoder"]), len(payloads["encoder_decoder"]))
    encoder, decoder = model_cache.resolve_pretrained_model("micro-sam", "encoder")
    assert Path(encoder).read_bytes() == payloads["encoder"]
    assert Path(decoder).read_bytes() == payloads["encoder_decoder"]
    model_cache.download_pretrained_model("micro-sam", "encoder")
    assert calls == ["encoder", "encoder_decoder"]
    Path(encoder).write_bytes(b"broken")
    assert not model_cache.list_pretrained_models("micro-sam")[0]["cached"]
    with pytest.raises(ValueError, match="Download"):
        model_cache.resolve_pretrained_model("micro-sam", "encoder")


@pytest.mark.parametrize("failure", ["cancel", "truncated", "checksum"])
def test_download_failures_never_publish_ready_models(
    catalogue: tuple, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _entries, payloads = catalogue
    cancelled = False

    def fetch(request: Any, **_kwargs: Any) -> Response:
        name = request.full_url.rsplit("/", 1)[1]
        data = payloads[name]
        return Response(
            data[:-1] if failure == "checksum" else data, size=len(data) + 1 if failure == "truncated" else None
        )

    def progress(*_args: Any) -> None:
        nonlocal cancelled
        cancelled = failure == "cancel"

    monkeypatch.setattr(model_cache, "urlopen", fetch)
    with pytest.raises((InterruptedError, OSError, ValueError)):
        model_cache.download_pretrained_model("micro-sam", "encoder", cancel=lambda: cancelled, task_progress=progress)
    assert not model_cache.list_pretrained_models("micro-sam")[0]["cached"]
    directory = model_cache.model_cache_directory("micro-sam")
    assert not list(directory.glob("*.part"))
    monkeypatch.setattr(
        model_cache, "urlopen", lambda request, **_kwargs: Response(payloads[request.full_url.rsplit("/", 1)[1]])
    )
    assert model_cache.download_pretrained_model("micro-sam", "encoder")["cached"]


def test_model_specific_requirements_without_importing_models(monkeypatch: pytest.MonkeyPatch) -> None:
    package = ModuleType("micro_sam")
    util = ModuleType("micro_sam.util")
    util.VIT_T_SUPPORT = False
    util.models = lambda: SimpleNamespace(
        registry={"vit_t": "hash", "vit_b": "hash", "vit_b_decoder": "hash"},
        urls={"vit_t": "https://example/t", "vit_b": "https://example/b", "vit_b_decoder": "https://example/d"},
    )
    monkeypatch.setitem(sys.modules, "micro_sam", package)
    monkeypatch.setitem(sys.modules, "micro_sam.util", util)
    entries = {entry["name"]: entry for entry in model_cache._registry("micro-sam")}
    assert "mobile_sam" in entries["vit_t"]["reason"]
    assert not entries["vit_b"]["reason"]
    assert len(entries["vit_b"]["assets"]) == 2
    models = ModuleType("cellpose.models")
    models.MODEL_NAMES = ["cpsam", "cpdino"]
    models._MODEL_URL = "https://example/"
    monkeypatch.setitem(sys.modules, "cellpose", ModuleType("cellpose"))
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    monkeypatch.setattr(model_cache, "find_spec", lambda _name: None)
    entries = {entry["name"]: entry for entry in model_cache._registry("cellpose")}
    assert "dinov3" in entries["cpdino"]["reason"]
    assert not entries["cpsam"]["reason"]


def test_retry_reuses_complete_encoder_after_decoder_failure(catalogue: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    _entries, payloads = catalogue
    calls = []
    interrupt_decoder = True

    def fetch(request: Any, **_kwargs: Any) -> Response:
        name = request.full_url.rsplit("/", 1)[1]
        calls.append(name)
        if interrupt_decoder and name == "encoder_decoder":
            raise OSError("Connection interrupted")
        return Response(payloads[name])

    monkeypatch.setattr(model_cache, "urlopen", fetch)
    with pytest.raises(OSError, match="interrupted"):
        model_cache.download_pretrained_model("micro-sam", "encoder")
    assert not model_cache.list_pretrained_models("micro-sam")[0]["cached"]
    interrupt_decoder = False
    model_cache.download_pretrained_model("micro-sam", "encoder")
    assert calls == ["encoder", "encoder_decoder", "encoder_decoder"]
    assert model_cache.list_pretrained_models("micro-sam")[0]["cached"]


def test_cache_receipt_verifies_assets_without_registry_hash(catalogue: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    entries, payloads = catalogue
    entries[0]["assets"] = [{**asset, "hash": ""} for asset in entries[0]["assets"]]
    calls = []

    def fetch(request: Any, **_kwargs: Any) -> Response:
        name = request.full_url.rsplit("/", 1)[1]
        calls.append(name)
        return Response(payloads[name])

    monkeypatch.setattr(model_cache, "urlopen", fetch)
    model_cache.download_pretrained_model("cellpose", "encoder")
    model_cache.download_pretrained_model("cellpose", "encoder")
    assert calls == ["encoder", "encoder_decoder"]


def test_catalogue_and_download_worker_jobs_do_not_require_project(
    catalogue: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancel = SimpleNamespace(is_set=lambda: False)
    result = run_job({"operation": "model_catalogue", "backend": "micro-sam"}, cancel, None)
    assert result[0]["name"] == "encoder"
    with pytest.raises(InterruptedError):
        run_job({"operation": "model_catalogue", "backend": "micro-sam"}, SimpleNamespace(is_set=lambda: True), None)
    monkeypatch.setattr(model_cache, "download_pretrained_model", lambda *args, **kwargs: {"cached": True})
    assert run_job({"operation": "download_model", "backend": "micro-sam", "model": "encoder"}, cancel, None)["cached"]


def test_model_descriptions_explain_names_and_requirements() -> None:
    assert "light microscopy" in model_cache.pretrained_model_help("micro-sam", "vit_b_lm")
    assert "Base" in model_cache.pretrained_model_help("micro-sam", "vit_b_lm")
    assert "mitochondria" in model_cache.pretrained_model_help("micro-sam", "vit_t_em_organelles")
    assert "low-contrast" in model_cache.pretrained_model_help("cellpose", "cpsam_v2")
    assert "dinov3" in model_cache.pretrained_model_help("cellpose", "cpdino-vitb")
    assert "no additional" in model_cache.pretrained_model_help("cellpose", "future_model")
