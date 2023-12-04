"""Transform utilities."""
from __future__ import annotations

import typing as ty

import numpy as np

if ty.TYPE_CHECKING:
    from skimage.transform import ProjectiveTransform


def compute_transform(src: np.ndarray, dst: np.ndarray, transform_type: str = "affine") -> ProjectiveTransform:
    """Compute transform."""
    from skimage.transform import estimate_transform

    if len(dst) != len(src):
        raise ValueError(f"The number of fixed and moving points is not equal. (moving={len(dst)}; fixed={len(src)})")
    return estimate_transform(transform_type, src, dst)


def transform_image(moving_image: np.ndarray, transform) -> np.ndarray:  # type: ignore[no-any-return]
    """Transform an image."""
    from skimage.transform import warp

    return warp(moving_image, transform, clip=False)


def combined_transform(
    image_size: tuple[int, int],
    image_spacing: tuple[float, float],
    rotation_angle: float | int,
    translation: tuple[float, float],
    flip_lr: bool = False,
) -> np.ndarray:
    """Combined transform."""
    tran = centered_translation_transform(translation)
    rot = centered_rotation_transform(image_size, image_spacing, rotation_angle)
    flip = np.eye(3)
    if flip_lr:
        flip = centered_horizontal_flip(image_size, image_spacing)
    return tran @ rot @ flip


def centered_translation_transform(
    translation: tuple[int, int],
) -> np.ndarray:
    """Centered translation transform."""
    translation = np.array(translation)
    transform = np.eye(3)
    transform[:2, 2] = translation
    return transform


def centered_rotation_transform(
    image_size: tuple[int, int],
    image_spacing: tuple[int, int],
    rotation_angle: float | int,
) -> np.ndarray:
    """Centered rotation transform."""
    angle = np.deg2rad(rotation_angle)

    sina = np.sin(angle)
    cosa = np.cos(angle)

    # build rot mat
    rot_mat = np.eye(3)
    rot_mat[0, 0] = cosa
    rot_mat[1, 1] = cosa
    rot_mat[1, 0] = sina
    rot_mat[0, 1] = -sina

    # recenter transform
    center_point = np.multiply(image_size, image_spacing) / 2
    center_point = np.append(center_point, 0)
    translation = center_point - np.dot(rot_mat, center_point)
    rot_mat[:2, 2] = translation[:2]

    return rot_mat


def centered_flip(
    image_size: tuple[int, int],
    image_spacing: tuple[int, int],
    direction: str,
) -> np.ndarray:
    """Centered flip transform."""
    angle = np.deg2rad(0)

    sina = np.sin(angle)
    cosa = np.cos(angle)

    # build rot mat
    rot_mat = np.eye(3)

    rot_mat[0, 0] = cosa
    rot_mat[1, 1] = cosa
    rot_mat[1, 0] = sina
    rot_mat[0, 1] = -sina

    if direction.lower() == "vertical":
        rot_mat[0, 0] = -1
    else:
        rot_mat[1, 1] = -1

    # recenter transform
    center_point = np.multiply(image_size, image_spacing) / 2
    center_point = np.append(center_point, 0)
    translation = center_point - np.dot(rot_mat, center_point)
    rot_mat[:2, 2] = translation[:2]

    return rot_mat


def centered_vertical_flip(
    image_size: tuple[int, int],
    image_spacing: tuple[int, int],
) -> np.ndarray:
    """Centered vertical flip transform."""
    return centered_flip(image_size, image_spacing, "vertical")


def centered_horizontal_flip(
    image_size: tuple[int, int],
    image_spacing: tuple[int, int],
) -> np.ndarray:
    """Centered horizontal flip transform."""
    return centered_flip(image_size, image_spacing, "horizontal")
