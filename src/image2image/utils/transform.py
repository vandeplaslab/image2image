"""Transform utilities."""
import typing as ty

import numpy as np

if ty.TYPE_CHECKING:
    from skimage.transform import ProjectiveTransform


def compute_transform(src: np.ndarray, dst: np.ndarray, transform_type: str = "affine") -> "ProjectiveTransform":
    """Compute transform."""
    from skimage.transform import estimate_transform

    if len(dst) != len(src):
        raise ValueError(f"The number of fixed and moving points is not equal. (moving={len(dst)}; fixed={len(src)})")
    return estimate_transform(transform_type, src, dst)


def transform_image(moving_image: np.ndarray, transform) -> np.ndarray:  # type: ignore[no-any-return]
    """Transform an image."""
    from skimage.transform import warp

    return warp(moving_image, transform, clip=False)


def centered_rotation_transform(
    image_size: ty.Tuple[int, int],
    image_spacing: ty.Tuple[int, int],
    rotation_angle: ty.Union[float, int],
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
    image_size: ty.Tuple[int, int],
    image_spacing: ty.Tuple[int, int],
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
    image_size: ty.Tuple[int, int],
    image_spacing: ty.Tuple[int, int],
) -> np.ndarray:
    """Centered vertical flip transform."""
    return centered_flip(image_size, image_spacing, "vertical")


def centered_horizontal_flip(
    image_size: ty.Tuple[int, int],
    image_spacing: ty.Tuple[int, int],
) -> np.ndarray:
    """Centered horizontal flip transform."""
    return centered_flip(image_size, image_spacing, "horizontal")
