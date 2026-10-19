"""Serial model, project and pipeline tests."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import SimpleITK as sitk
import tifffile
from click.testing import CliRunner

from image2image.models.serial import SerialSettings
from image2image.models.wsiprep import Registration, RegistrationImage
from image2image.serial_cli import serial
from image2image.utils.serial import (
    build_serial_registration,
    inspect_sections,
    load_serial_project,
    reconstruct_serial,
    register_serial,
    registration_fingerprint,
    save_serial_project,
)


def make_project(tmp_path: Path) -> Registration:
    registration = Registration(serial=SerialSettings(spacing_um=5))
    for index in (10, 2, 1):
        path = tmp_path / f"section{index}.ome.tiff"
        pixels = np.full((2, 24, 28), index, dtype=np.uint16)
        tifffile.imwrite(
            path,
            pixels,
            metadata={"axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "Channel": {"Name": ["DAPI", "FITC"]}},
        )
        registration.images[path.stem] = RegistrationImage(path=path, key=path.stem)
    registration.serial.synchronize(registration)
    return registration


def add_identity_registration(project: Registration, run_dir: Path) -> None:
    """Attach known transforms for testing reconstruction independently of alignment."""
    settings = project.serial
    fingerprint = registration_fingerprint(project)
    run_dir.mkdir()
    transforms = {}
    for key in settings.ordered_keys(project):
        transform_path = run_dir / f"{key}.h5"
        sitk.WriteTransform(sitk.Transform(2, sitk.sitkIdentity), str(transform_path))
        transforms[str(settings.section_numbers[key])] = transform_path.name
    (run_dir / "transforms.json").write_text(json.dumps({"fingerprint": fingerprint, "transforms": transforms}))
    settings.result_dir = str(run_dir)
    settings.registered_fingerprint = fingerprint


def test_order_and_neighbor_paths(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    settings = project.serial
    keys = settings.ordered_keys(project)
    assert keys == ["section1.ome", "section2.ome", "section10.ome"]
    assert settings.reference == keys[1]
    settings.section_numbers = dict(zip(keys, [1, 4, 8], strict=False))
    assert settings.registration_paths(project) == [(keys[0], keys[1], None), (keys[2], keys[1], None)]
    settings.reference = keys[0]
    assert settings.registration_paths(project) == [(keys[1], keys[0], None), (keys[2], keys[0], keys[1])]
    project.images[keys[1]].keep = False
    assert settings.registration_paths(project) == [(keys[2], keys[0], None)]
    assert settings.section_numbers[keys[1]] == 4


def test_numbering_uses_filenames_not_model_keys(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    project.images = {str(index): model for index, model in enumerate(project.images.values())}
    project.serial = SerialSettings()
    project.serial.synchronize(project)
    assert [project.images[key].path.name for key in project.serial.ordered_keys(project)] == [
        "section1.ome.tiff",
        "section2.ome.tiff",
        "section10.ome.tiff",
    ]


def test_zarr_sections_and_chunk_invalidation(tmp_path: Path) -> None:
    import zarr
    from image2image_io.volume import open_volume
    from image2image_io.writers import write_ome_zarr_from_array

    project = make_project(tmp_path)
    for key, model in project.images.items():
        pixels = tifffile.imread(model.path)
        model.path = write_ome_zarr_from_array(
            tmp_path / f"{key}.zarr", None, pixels, resolution=0.5, channel_names=["DAPI", "FITC"], tile_size=16
        )
    before = registration_fingerprint(project)
    source = zarr.open_group(next(iter(project.images.values())).path, mode="a")
    source["0"][0, 0, 0] = 99
    assert registration_fingerprint(project) != before
    add_identity_registration(project, tmp_path / "registered")
    path = tmp_path / "stack.i2wsiprep.json"
    save_serial_project(path, project)
    volume = open_volume(reconstruct_serial(path, tmp_path / "volume.ome.zarr"))
    assert volume.levels[0][0, 2, 0, 0].compute() == 99
    assert volume.channel_names == ("DAPI", "FITC")


@pytest.mark.parametrize("suffix", ["json", "toml"])
def test_project_roundtrip(tmp_path: Path, suffix: str) -> None:
    project = make_project(tmp_path)
    project.serial.section_numbers = dict(zip(project.serial.ordered_keys(project), [3, 5, 8], strict=False))
    project.images["section10.ome"].keep = False
    path = tmp_path / f"stack.i2wsiprep.{suffix}"
    save_serial_project(path, project)
    loaded = load_serial_project(path)
    assert loaded.to_dict() == project.to_dict()
    assert registration_fingerprint(loaded) == registration_fingerprint(project)
    legacy = Registration.model_validate({key: value for key, value in project.to_dict().items() if key != "serial"})
    assert legacy.serial is None
    assert "serial" not in legacy.to_dict()


def test_validation_and_invalidation(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    with pytest.raises(ValueError):
        project.serial.spacing_um = 0
    with pytest.raises(ValueError):
        project.serial.section_numbers = dict.fromkeys(project.images, 1)
    fingerprint = registration_fingerprint(project)
    project.serial.canvas_xywh = (-10, -10, 100, 100)
    project.serial.spacing_um = 10
    assert registration_fingerprint(project) == fingerprint
    project.images[project.serial.reference].rotate = 45
    assert registration_fingerprint(project) != fingerprint
    project.images[project.serial.reference].keep = False
    with pytest.raises(ValueError, match="reference"):
        inspect_sections(project)


def test_build_registration_graph(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    project.serial.registration_channel = 1
    backend = build_serial_registration(project, tmp_path / "results", inspect_sections(project))
    assert len(backend.registration_nodes) == 2
    assert all(edge["modalities"]["target"] == "section_2" for edge in backend.registration_nodes)
    assert all(edge["params"] == ["rigid"] for edge in backend.registration_nodes)
    assert all(modality.preprocessing.channel_indices == [1] for modality in backend.modalities.values())


def test_reconstruct_native_channels_and_gaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from image2image_io import readers
    from image2image_io.readers import BaseReader
    from image2image_io.volume import open_volume

    original_reader = readers.get_simple_reader
    active: set[int] = set()
    counts: list[int] = []

    def read_section(path: Path, **kwargs: Any) -> BaseReader:
        reader = original_reader(path, **kwargs)
        active.add(id(reader))
        counts.append(len(active))
        close = reader.close

        def close_section() -> None:
            active.discard(id(reader))
            close()

        monkeypatch.setattr(reader, "close", close_section)
        return reader

    monkeypatch.setattr(readers, "get_simple_reader", read_section)
    project = make_project(tmp_path)
    settings = project.serial
    keys = settings.ordered_keys(project)
    settings.section_numbers = dict(zip(keys, [1, 3, 4], strict=False))
    project.images[keys[-1]].keep = False
    add_identity_registration(project, tmp_path / "registered")
    path = tmp_path / "stack.i2wsiprep.json"
    save_serial_project(path, project)
    output = reconstruct_serial(path, tmp_path / "volume.ome.zarr")
    volume = open_volume(output)
    assert volume.levels[0].shape == (2, 4, 24, 28)
    assert volume.spacing == (5, 0.5, 0.5)
    assert volume.present == (True, False, True, False)
    assert load_serial_project(volume.provenance["project_snapshot"]).to_dict() == project.to_dict()
    np.testing.assert_array_equal(volume.levels[0][:, 0].compute(), 1)
    np.testing.assert_array_equal(volume.levels[0][:, 2].compute(), 2)
    assert load_serial_project(path).serial.volume_path == str(output)
    assert max(counts) == 1
    assert not active
    settings.preset = "affine"
    save_serial_project(path, project)
    with pytest.raises(ValueError, match="stale"):
        reconstruct_serial(path, tmp_path / "stale.ome.zarr")


def test_cli_error_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text('{"tool": "wsiprep"}')
    result = CliRunner().invoke(serial, ["reconstruct", "--project", str(path), "--output", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "Serial Sections settings" in result.output


def test_prepared_reference_canvas_and_native_intensities(tmp_path: Path) -> None:
    from image2image_io.volume import open_volume

    yy, xx = np.mgrid[:48, :64]
    signal = 20000 * np.exp(-((xx - 25) ** 2 + (yy - 18) ** 2) / 80)
    pixels = np.stack([signal, signal / 2 + xx * 10]).astype(np.uint16)
    project = Registration(serial=SerialSettings())
    for index in range(2):
        path = tmp_path / f"section{index}.ome.tiff"
        tifffile.imwrite(path, pixels, metadata={"axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5})
        project.images[path.stem] = RegistrationImage(
            key=path.stem, path=path, rotate=90, flip_lr=True, translate_x=6, translate_y=4
        )
    project.serial.synchronize(project)
    path = tmp_path / "prepared.i2wsiprep.json"
    save_serial_project(path, project)
    register_serial(path)
    volume = open_volume(reconstruct_serial(path, tmp_path / "prepared.ome.zarr"))
    expected = np.rot90(np.flip(pixels, axis=2), axes=(1, 2))
    reference = volume.levels[0][:, 0].compute()
    np.testing.assert_allclose(reference[:, 8:72, 12:60], expected, atol=1)
    assert not reference[:, :8].any()
    assert not reference[:, :, :12].any()
    moving = volume.levels[0][:, 1].compute()
    assert np.mean(np.abs(reference.astype(float) - moving)) < 50


@pytest.mark.parametrize(
    ("rgb", "preset"), [(False, "rigid"), (True, "rigid"), (False, "affine"), (False, "deformable")]
)
def test_synthetic_registration_and_reconstruction(tmp_path: Path, rgb: bool, preset: str) -> None:
    """Run real Elastix presets, including composed paths on both sides."""
    from image2image_io.volume import open_volume

    yy, xx = np.mgrid[:192, :224]
    signal = sum(
        amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / radius**2)
        for x, y, radius, amplitude in [
            (54, 70, 15, 20000),
            (150, 80, 23, 35000),
            (88, 142, 18, 25000),
            (160, 130, 8, 50000),
        ]
    )
    pixels = np.stack([signal, signal / 2 + 100]).astype(np.uint16)
    if rgb:
        pixels = np.stack([255 - signal / 220, 245 - signal / 260, 235 - signal / 300]).astype(np.uint8)
    project = Registration(serial=SerialSettings(spacing_um=7.5, preset=preset))
    for index, shift in enumerate([-6, -3, 0, 3, 6]):
        path = tmp_path / f"section{index}.ome.tiff"
        source = np.roll(pixels, (shift, -shift), axis=(1, 2))
        if index == 4:
            source = np.flip(source, axis=2)
        if index == 0:
            source = np.rot90(source, axes=(1, 2))
        metadata = {"axes": "YXS" if rgb else "CYX", "PhysicalSizeX": 0.7, "PhysicalSizeY": 0.7}
        if not rgb:
            metadata["Channel"] = {"Name": ["DAPI", "FITC"]}
        tifffile.imwrite(
            path,
            np.moveaxis(source, 0, -1) if rgb else source,
            photometric="rgb" if rgb else "minisblack",
            metadata=metadata,
        )
        project.images[path.stem] = RegistrationImage(
            key=path.stem, path=path, flip_lr=index == 4, rotate=270 if index == 0 else 0
        )
    project.serial.synchronize(project)
    project.serial.section_numbers = dict(zip(project.serial.ordered_keys(project), [1, 3, 6, 9, 10], strict=True))
    path = tmp_path / "synthetic.i2wsiprep.json"
    save_serial_project(path, project)
    register_serial(path)
    volume = open_volume(reconstruct_serial(path, tmp_path / "synthetic.ome.zarr"))
    assert volume.present == (True, False, True, False, False, True, False, False, True, True)
    assert volume.channel_names == (("R", "G", "B") if rgb else ("DAPI", "FITC"))
    assert volume.spacing == (7.5, 0.7, 0.7)
    for z in [0, 2, 5, 8, 9]:
        actual = volume.levels[0][:, z].compute()
        assert actual.dtype == pixels.dtype
        # Ignore the empty border exposed by translation of a brightfield image.
        assert np.mean(np.abs(actual[:, 12:-12, 12:-12].astype(float) - pixels[:, 12:-12, 12:-12])) < (
            3 if rgb else 500
        )
