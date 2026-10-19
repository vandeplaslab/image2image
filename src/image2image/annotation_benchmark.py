"""Reproducible Annotate storage/render benchmark with isolated, measured workloads."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

from image2image.models.annotation import AnnotationClass, AnnotationObject, AnnotationSlide, SlideGeometry
from image2image.utils.annotation.render import AnnotationRenderer
from image2image.utils.annotation.store import AnnotationStore


def benchmark(count: int, samples: int, directory: Path) -> dict:
    """Measure the real public store/renderer against seeded mixed polygon geometry."""
    import psutil

    process = psutil.Process()
    rng = np.random.default_rng(42)
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    shape = (rows * 32, columns * 32)
    project = directory / "benchmark.i2annotate"
    baseline_rss = process.memory_info().rss
    start = perf_counter()
    with AnnotationStore(project, create=True) as store:
        slide = AnnotationSlide(
            name="Synthetic object grid", path="synthetic-no-image", geometry=SlideGeometry(shape=shape)
        )
        store.save_slide(slide)
        classes = [
            AnnotationClass(name=name, color=color)
            for name, color in (("A", "#ef5350"), ("B", "#42a5f5"), ("C", "#66bb6a"))
        ]
        for cls in classes:
            store.save_class(cls)
        set_id = store.create_set(slide.id, "Mixed polygon benchmark", staging=True)
        batch = []
        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        for index in range(count):
            x, y = (index % columns) * 32 + 16, (index // columns) * 32 + 16
            radii = rng.uniform(6, 9, 20)
            ring = np.c_[x + radii * np.cos(angles), y + radii * np.sin(angles)]
            geometry = Polygon(ring)
            if index % 20 == 0:
                geometry = Polygon(ring, [box(x + 1, y + 1, x + 3, y + 3).exterior])
            elif index % 20 == 1:
                geometry = MultiPolygon([geometry, box(x + 11, y + 11, x + 13, y + 13)])
            batch.append(AnnotationObject(set_id, index + 1, geometry, classes[index % 3].id))
            if len(batch) == 1000:
                store.append_staged(set_id, batch)
                batch.clear()
            if (index + 1) % 100000 == 0:
                print(f"Generated {index + 1:,}/{count:,} objects", flush=True)
        if batch:
            store.append_staged(set_id, batch)
        store.publish_set(set_id)
        if not (store.count(set_id) == count):
            raise RuntimeError("Benchmark correctness check failed: store.count(set_id) == count")
        build_seconds = perf_counter() - start
        store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    reopen_start = perf_counter()
    with AnnotationStore(project) as store:
        reopen_seconds = perf_counter() - reopen_start
        selected_ids = rng.integers(0, count, size=samples)
        selection_ms = []
        for index in selected_ids:
            xy = (int(index % columns) * 32 + 16, int(index // columns) * 32 + 16)
            start = perf_counter()
            obj = store.hit_test(slide.id, xy, set_id=set_id)
            selection_ms.append((perf_counter() - start) * 1000)
            if not (obj is not None and obj.object_id == int(index) + 1):
                raise RuntimeError(
                    "Benchmark correctness check failed: obj is not None and obj.object_id == int(index) + 1"
                )
        renderer = AnnotationRenderer(project, max_bytes=128 * 1024 * 1024)
        viewport = (0, 0, min(1024, shape[1]), min(1024, shape[0]))
        start = perf_counter()
        first_frame = renderer.render(slide.id, viewport, set_id=set_id)
        first_frame_ms = (perf_counter() - start) * 1000
        if not (first_frame.rgba[..., 3].any()):
            raise RuntimeError("Benchmark correctness check failed: first_frame.rgba[..., 3].any()")
        start = perf_counter()
        overview = renderer.render(slide.id, (0, 0, shape[1], shape[0]), set_id=set_id)
        overview_ms = (perf_counter() - start) * 1000
        if not (overview.rgba[..., 3].any()):
            raise RuntimeError("Benchmark correctness check failed: overview.rgba[..., 3].any()")
        warm_ms = []
        for _ in range(samples):
            start = perf_counter()
            frame = renderer.render(slide.id, viewport, set_id=set_id)
            warm_ms.append((perf_counter() - start) * 1000)
            if not (frame.rgba.shape == first_frame.rgba.shape):
                raise RuntimeError("Benchmark correctness check failed: frame.rgba.shape == first_frame.rgba.shape")
        cache_bytes = renderer.cache_bytes
        if not (cache_bytes <= renderer.max_bytes):
            raise RuntimeError("Benchmark correctness check failed: cache_bytes <= renderer.max_bytes")
        resident_rss = process.memory_info().rss
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        except ImportError:
            peak_bytes = None
    result = {
        "count": count,
        "seed": 42,
        "vertices_per_exterior": 20,
        "holes_fraction": 0.05,
        "multipart_fraction": 0.05,
        "shape_yx": shape,
        "build_and_journal_seconds": build_seconds,
        "reopen_seconds": reopen_seconds,
        "first_frame_after_import_ms": first_frame_ms,
        "first_overview_ms": overview_ms,
        "selection_ms": selection_ms,
        "cached_overlay_ms": warm_ms,
        "selection_p95_ms": float(np.percentile(selection_ms, 95)),
        "cached_overlay_p95_ms": float(np.percentile(warm_ms, 95)),
        "baseline_rss_bytes": baseline_rss,
        "resident_rss_bytes": resident_rss,
        "peak_rss_bytes": peak_bytes,
        "cache_bytes": cache_bytes,
        "cache_limit_bytes": renderer.max_bytes,
        "database_bytes": (project / "objects.sqlite").stat().st_size,
        "notes": "Synthetic mixed polygons, real store and renderer; no source image decoding or Qt painting. First render follows import, so OS disk caches may be warm. No timing samples discarded.",
    }
    result["targets_met"] = result["selection_p95_ms"] < 100 and result["cached_overlay_p95_ms"] < 250
    return result


def main() -> None:
    """Run one million and two million objects in separate processes and save raw results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", default="1000000,2000000")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("annotation-benchmark.json"))
    parser.add_argument("--single", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    if args.single is not None:
        with TemporaryDirectory(prefix="i2i-annotation-benchmark-") as directory:
            result = benchmark(args.single, args.samples, Path(directory))
            args.output.write_text(json.dumps(result, indent=2))
        return
    import psutil

    results = []
    for count in [int(value) for value in args.counts.split(",")]:
        if count < 1:
            parser.error("counts must be positive")
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            subprocess.run(  # noqa: S603 - fixed module and interpreter; validated numeric arguments
                [  # noqa: S603 - fixed module, local interpreter, validated integer arguments
                    sys.executable,
                    "-m",
                    "image2image.annotation_benchmark",
                    "--single",
                    str(count),
                    "--samples",
                    str(args.samples),
                    "--output",
                    str(output),
                ],
                check=True,
            )  # noqa: S603
            results.append(json.loads(output.read_text()))
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote raw measurements to {args.output}", flush=True)


if __name__ == "__main__":
    main()
