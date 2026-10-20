"""Persistent spawned worker entrypoint; model imports never run on the Qt thread."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any


def run_job(job: dict, cancel: Any, progress: Any, task_progress: Any = None) -> Any:
    """Execute one serializable project operation in a spawned process."""
    from image2image.models.annotation import CoordinateSpec, FeatureSettings, FineTuneSettings, PredictionRequest
    from image2image.utils.annotation.finetune import fine_tune_model, list_finetuned_models
    from image2image.utils.annotation.io import (
        check_cancel,
        export_geojson,
        export_labels,
        import_geojson,
        import_labels,
    )
    from image2image.utils.annotation.ml import predict, train_classifier
    from image2image.utils.annotation.model_cache import download_pretrained_model, list_pretrained_models
    from image2image.utils.annotation.store import AnnotationStore

    operation = job["operation"]
    if operation == "model_catalogue":
        check_cancel(cancel.is_set)
        catalogue = list_pretrained_models(job["backend"])
        if job.get("project"):
            catalogue.extend(
                finetuned_entry(
                    entry,
                    next((base["reason"] for base in catalogue if base["name"] == entry["settings"]["model"]), ""),
                )
                for entry in list_finetuned_models(Path(job["project"]), job["backend"])
            )
        check_cancel(cancel.is_set)
        return catalogue
    if operation == "download_model":
        return download_pretrained_model(
            job["backend"], job["model"], cancel=cancel.is_set, task_progress=task_progress, download_id=job.get("id")
        )
    with AnnotationStore(job["project"]) as store:
        if operation == "combine_sets":
            return store.combine_sets(job["set_ids"], job["name"], cancel=cancel.is_set, task_progress=task_progress)
        kws = {"cancel": cancel.is_set, "progress": progress}
        if operation == "train":
            return str(train_classifier(store, job["name"], FeatureSettings.model_validate(job["settings"]), **kws))
        if operation == "finetune":
            return fine_tune_model(
                store,
                job["name"],
                FineTuneSettings.model_validate(job["settings"]),
                cancel=cancel.is_set,
                task_progress=task_progress,
            )
        if operation == "predict":
            if task_progress is not None:
                kws["progress"] = None
            return predict(
                store, PredictionRequest.model_validate(job["request"]), task_progress=task_progress, **kws
            ).model_dump()
        if operation in ("undo", "redo"):
            return getattr(store, operation)()
        if operation == "review":
            return store.review(job["set_id"], accept=job["accept"], keys=job.get("keys"), cancel=cancel.is_set)
        coordinates = CoordinateSpec.model_validate(job["coordinates"])
        if operation == "import":
            importer = import_geojson if job["format"] == "geojson" else import_labels
            return importer(store, job["slide_id"], job["path"], coordinates, **kws)
        if operation == "export":
            exporter = export_geojson if job["format"] == "geojson" else export_labels
            if job["format"] != "geojson":
                kws["mapped_ids"] = job.get("mapped_ids", False)
            return str(exporter(store, job["set_id"], job["path"], coordinates, **kws))
        raise ValueError(f"Unknown annotation worker operation: {operation}")


def finetuned_entry(metadata: dict[str, Any], reason: str = "") -> dict[str, Any]:
    """Represent a project checkpoint alongside managed pretrained model entries."""
    return {
        "name": metadata["checkpoint"],
        "display_name": metadata["name"] + " (fine-tuned)",
        "path": metadata["checkpoint"],
        "cached": True,
        "reason": reason,
        "custom": metadata,
    }


def worker_loop(incoming: Any, outgoing: Any, cancel: Any) -> None:
    """Keep model caches warm across sequential jobs; report errors as plain records."""
    while True:
        job = incoming.get()
        if job is None:
            return
        job_id = job["id"]

        def progress(count: int, message: str, job_id: str = job_id) -> None:
            outgoing.put({"id": job_id, "kind": "progress", "count": count, "message": message})

        def task_progress(phase: str, completed: int, total: int | None, message: str, job_id: str = job_id) -> None:
            outgoing.put(
                {
                    "id": job_id,
                    "kind": "progress",
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "message": message,
                }
            )

        try:
            result = run_job(job, cancel, progress, task_progress)
            outgoing.put({"id": job_id, "kind": "finished", "result": result})
        except Exception as exc:  # noqa: BLE001 - process boundary must serialize backend failures
            outgoing.put({"id": job_id, "kind": "error", "message": str(exc), "traceback": traceback.format_exc()})
