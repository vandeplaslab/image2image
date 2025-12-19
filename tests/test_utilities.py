"""Test image2image utilities."""

import numpy as np
import pytest
from image2image.utils.utilities import (
    ensure_list,
    extract_extension,
    extract_number,
    get_groups,
    get_random_hex_color,
    pad_str,
    round_to_half,
)


def test_ensure_list():
    """Test ensure_list."""
    assert ensure_list("a") == ["a"]
    assert ensure_list(["a"]) == ["a"]
    assert ensure_list(1) == [1]
    assert ensure_list(None) == [None]


def test_pad_str(monkeypatch):
    """Test pad_str."""
    from koyo import system
    
    # Mock IS_MAC to True
    monkeypatch.setattr(system, "IS_MAC", True)
    assert pad_str("hello") == "hello"

    # Mock IS_MAC to False
    monkeypatch.setattr(system, "IS_MAC", False)
    assert pad_str("hello") == '"hello"'


def test_extract_number():
    """Test extract_number."""
    assert extract_number("image_001.tif", "image_") == "001"
    assert extract_number("data123.dat", "data") == "123"
    assert extract_number("no_number.txt", "data") is None


def test_extract_extension():
    """Test extract_extension."""
    formats = "*.tif;;*.png;;*.jpg"
    exts = extract_extension(formats)
    assert ".tif" in exts
    assert ".png" in exts
    assert ".jpg" in exts
    assert len(exts) == 3


def test_round_to_half():
    """Test round_to_half."""
    arr = np.array([0.1, 0.4, 0.6, 0.9])
    rounded = round_to_half(arr)
    expected = np.array([0.0, 0.5, 0.5, 1.0])
    np.testing.assert_array_equal(rounded, expected)


def test_get_groups():
    """Test get_groups."""
    filenames = ["img_s1.tif", "img_s1_b.tif", "img_s2.tif", "other.txt"]
    groups = get_groups(filenames, "img_s")
    # Groups logic depends on extract_number
    # img_s1 -> group 1
    # img_s2 -> group 2
    # other.txt -> no group
    
    # Check logic of get_groups:
    # it uses extract_number(filename, keyword)
    # extract_number("img_s1.tif", "img_s") -> "1"
    
    assert "1" in groups
    assert "img_s1.tif" in groups["1"]
    assert "img_s1_b.tif" in groups["1"]
    assert "2" in groups
    assert "img_s2.tif" in groups["2"]
    assert "no group" in groups
    assert "other.txt" in groups["no group"]


def test_get_random_hex_color():
    """Test get_random_hex_color."""
    color = get_random_hex_color()
    assert color.startswith("#")
    assert len(color) == 7
    # Check it is valid hex
    int(color[1:], 16)
