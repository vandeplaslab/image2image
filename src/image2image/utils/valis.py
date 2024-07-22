"""Valis."""

from __future__ import annotations

import typing as ty

from koyo.secret import hash_parameters

if ty.TYPE_CHECKING:
    from image2image_reg.models import Preprocessing


def guess_preprocessing(reader, valis: bool = False) -> Preprocessing:
    """Guess pre-processing."""
    from image2image_reg.models import Preprocessing

    if reader.is_rgb:
        return Preprocessing.brightfield(valis=valis)
    return Preprocessing.fluorescence(valis=valis)


def hash_preprocessing(preprocessing: Preprocessing, pyramid: int = -1) -> str:
    """Hash preprocessing."""
    return hash_parameters(**preprocessing.to_dict(), pyramid=pyramid, n_in_hash=6)
