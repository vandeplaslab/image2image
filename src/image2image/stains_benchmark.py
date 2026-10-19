"""Reproducible stain processing benchmarks with real tiled reads and complete exports."""

from __future__ import annotations

import argparse
import json
import os
import platform
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter
from typing import Any

import numpy as np
import psutil
import tifffile

from image2image.models.stains import StainRecipe, stain_preset
from image2image.utils.stain_io import export_stains, fit_source, inspect_stain_file, preview_stains
from image2image.utils.stains import separate_stains, stain_matrix


def benchmark(size: int = 2048, repeats: int = 3, output_format: str = "ome.tiff") -> dict[str, Any]:
    """Measure seeded stain mixtures, fitting, first/warm calls, export throughput, and process RSS."""
    if size < 512 or repeats < 1 or output_format not in ("ome.tiff", "ome.zarr"):
        raise ValueError("Use size >= 512, repeats >= 1, and ome.tiff or ome.zarr.")
    profile = stain_preset()
    basis = stain_matrix(profile)
    process = psutil.Process()
    stop = Event()
    baseline = process.memory_info().rss
    peak = [baseline]

    def monitor() -> None:
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor_thread = Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        with TemporaryDirectory(prefix="i2i-stain-benchmark-") as temporary:
            directory = Path(temporary)
            path = directory / "synthetic.ome.tiff"
            rng = np.random.default_rng(2026)

            def tiles() -> Iterator[np.ndarray]:
                for y in range(0, size, 512):
                    for x in range(0, size, 512):
                        coefficients = np.zeros((512, 512, 3), dtype=np.float32)
                        coefficients[..., :2] = rng.gamma(0.5, 0.6, (512, 512, 2))
                        yy, xx = np.ogrid[y : y + 512, x : x + 512]
                        tissue = (np.sin(yy / 170) + np.cos(xx / 213)) > -0.9
                        coefficients *= tissue[..., None]
                        pixels = np.rint(np.exp(-coefficients @ basis) * 255)
                        # Nontrivial noise and spatial structure, including bright background.
                        yield np.clip(pixels + rng.integers(-2, 3, pixels.shape), 0, 255).astype(np.uint8)

            start = perf_counter()
            with tifffile.TiffWriter(path, ome=True, bigtiff=True) as writer:
                writer.write(
                    tiles(),
                    shape=(size, size, 3),
                    dtype=np.uint8,
                    tile=(512, 512),
                    photometric="rgb",
                    compression="deflate",
                    metadata={"axes": "YXS"},
                )
            generation = perf_counter() - start
            source = inspect_stain_file(path)[0]
            recipe = StainRecipe(reference=source, export_normalized=True)
            probe = rng.integers(1, 256, (512, 512, 3), dtype=np.uint8)
            start = perf_counter()
            actual = separate_stains(probe, profile)
            first_call = perf_counter() - start
            np.testing.assert_allclose(actual, separate_stains(probe, profile, compiled=False), atol=2e-6)
            kernels = []
            for _ in range(repeats):
                start = perf_counter()
                separate_stains(probe, profile)
                kernels.append(perf_counter() - start)
            start = perf_counter()
            fit = fit_source(source, recipe)
            fitting = perf_counter() - start
            cache = {}
            start = perf_counter()
            preview_stains(source, recipe, region=(0, 0, 512, 512), cache=cache)
            cold_preview = perf_counter() - start
            previews = []
            for _ in range(repeats):
                start = perf_counter()
                preview_stains(source, recipe, region=(0, 0, 512, 512), cache=cache)
                previews.append(perf_counter() - start)
            before_export = process.memory_info().rss
            peak[0] = before_export
            start = perf_counter()
            outputs = export_stains(source, recipe, directory / "out", output_format=output_format)
            exporting = perf_counter() - start
            export_peak = max(peak[0], process.memory_info().rss)
            output_bytes = sum(
                sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
                if path.is_dir()
                else path.stat().st_size
                for path in outputs
            )
            return {
                "hardware": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "logical_cpus": os.cpu_count(),
                },
                "fixture": {
                    "kind": "seeded noisy synthetic stain mixtures with spatially varying tissue/background",
                    "size": [size, size],
                    "dtype": "uint8",
                    "generation_seconds": generation,
                },
                "numba_first_call_seconds": first_call,
                "numba_cache_directory": os.environ.get(
                    "NUMBA_CACHE_DIR", "default; first call may load an existing disk cache"
                ),
                "kernel_median_seconds_512px": float(np.median(kernels)),
                "kernel_megapixels_per_second": 512**2 / float(np.median(kernels)) / 1e6,
                "fitting_seconds_including_sampling": fitting,
                "fitting_tissue_samples": fit.sample_count,
                "preview_first_seconds": cold_preview,
                "preview_warm_median_seconds": float(np.median(previews)),
                "preview_definition": "native 512-pixel crop plus independent reference overview; warm runs reuse fits",
                "export_format": output_format,
                "export_seconds_including_pyramids_and_compression": exporting,
                "export_megapixels_per_second": size**2 / exporting / 1e6,
                "export_bytes": output_bytes,
                "export_baseline_rss_mib": before_export / 1024**2,
                "export_peak_rss_mib": export_peak / 1024**2,
                "export_rss_growth_mib": (export_peak - before_export) / 1024**2,
                "rss_sampling_interval_seconds": 0.01,
            }
    finally:
        stop.set()
        monitor_thread.join()


def main() -> None:
    """Run a local benchmark without retaining generated slide data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--format", choices=("ome.tiff", "ome.zarr"), default="ome.tiff")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(benchmark(args.size, args.repeats, args.format), indent=2)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
