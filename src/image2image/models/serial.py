"""Serial-section geometry and registration settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from natsort import natsorted
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from image2image.models.wsiprep import Registration


class SerialSettings(BaseModel):
    """Physical section positions are independent of table and registration order."""

    model_config = ConfigDict(validate_assignment=True, allow_inf_nan=False)
    section_numbers: dict[str, int] = Field(default_factory=dict)
    spacing_um: float = Field(1.0, gt=0)
    reference: str | None = None
    preset: Literal["rigid", "affine", "deformable"] = "rigid"
    registration_channel: int = Field(0, ge=0)
    registration_downsample: int = Field(1, ge=1)
    max_registration_pixels: int = Field(4_000_000, gt=0)
    canvas_xywh: tuple[float, float, float, float] | None = None
    output_spacing_um: float | None = Field(None, gt=0)
    result_dir: str | None = None
    registered_fingerprint: str | None = None
    volume_path: str | None = None

    @field_validator("section_numbers", mode="before")
    @classmethod
    def validate_numbers(cls, value: dict[str, int]) -> dict[str, int]:
        """Reject duplicate, fractional, and nonpositive section numbers."""
        if any(type(number) is not int or number <= 0 for number in value.values()):
            raise ValueError("Section numbers must be positive integers.")
        if len(set(value.values())) != len(value):
            raise ValueError("Section numbers must be unique.")
        return value

    @field_validator("canvas_xywh")
    @classmethod
    def validate_canvas(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        """Check finite physical canvas bounds."""
        if value is not None and (not np.all(np.isfinite(value)) or min(value[2:]) <= 0):
            raise ValueError("Canvas width and height must be positive and all bounds finite.")
        return value

    def synchronize(self, registration: Registration) -> None:
        """Number newly added images without renumbering existing sections."""
        numbers = {key: number for key, number in self.section_numbers.items() if key in registration.images}
        next_number = max(numbers.values(), default=0) + 1
        for key in natsorted(registration.images, key=lambda key: registration.images[key].path.name):
            if key not in numbers:
                numbers[key] = next_number
                next_number += 1
        self.section_numbers = numbers
        included = self.ordered_keys(registration)
        if self.reference is None and included:
            self.reference = included[(len(included) - 1) // 2]

    def ordered_keys(self, registration: Registration, included_only: bool = True) -> list[str]:
        """Return physical section order, optionally retaining excluded sections."""
        return sorted(
            (key for key in registration.images if not included_only or registration.images[key].keep),
            key=lambda key: self.section_numbers[key],
        )

    def validate_sections(self, registration: Registration) -> None:
        """Validate a complete stack before registration or reconstruction."""
        if set(self.section_numbers) != set(registration.images):
            raise ValueError("Every image must have a section number.")
        self.validate_numbers(self.section_numbers)
        included = self.ordered_keys(registration)
        if len(included) < 2:
            raise ValueError("Include at least two sections.")
        if self.reference not in included:
            raise ValueError("Choose a reference section that is included in the stack.")

    def registration_paths(self, registration: Registration) -> list[tuple[str, str, str | None]]:
        """Build adjacent paths toward the reference, nearest neighbors first."""
        self.validate_sections(registration)
        keys = self.ordered_keys(registration)
        reference = self.reference
        index = keys.index(reference)
        paths = []
        for side in (list(reversed(keys[:index])), keys[index + 1 :]):
            neighbor = reference
            for key in side:
                paths.append((key, reference, None if neighbor == reference else neighbor))
                neighbor = key
        return paths
