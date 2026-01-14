"""Common schema types for ML model inputs and outputs."""

import base64
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator


class Base64Data(BaseModel):
    """Base64-encoded binary data with media type.

    Useful for sending images or other binary data in JSON requests.
    """

    data: str
    media_type: str = "application/octet-stream"

    @field_validator("data")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v)
        except Exception as e:
            raise ValueError("Invalid base64") from e
        return v

    def decode(self) -> bytes:
        return base64.b64decode(self.data)


class BBox(BaseModel):
    """Bounding box with (x1, y1) as top-left and (x2, y2) as bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x2")
    @classmethod
    def x2_gt_x1(cls, v: float, info: Any) -> float:
        if "x1" in info.data and v <= info.data["x1"]:
            raise ValueError("x2 must be > x1")
        return v

    @field_validator("y2")
    @classmethod
    def y2_gt_y1(cls, v: float, info: Any) -> float:
        if "y1" in info.data and v <= info.data["y1"]:
            raise ValueError("y2 must be > y1")
        return v

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class Label(BaseModel):
    """Classification label with confidence score."""

    name: str
    confidence: float = Field(..., ge=0, le=1)


class Vector(BaseModel):
    """Embedding vector with float values."""

    values: list[float] = Field(..., min_length=1)

    @property
    def dim(self) -> int:
        """Dimensionality of the vector."""
        return len(self.values)


class Span(BaseModel):
    """Text span with character offsets, optional label and score."""

    text: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    label: str | None = None
    score: float | None = Field(None, ge=0, le=1)


Prob = Annotated[float, Field(ge=0, le=1)]
"""Probability value constrained to [0, 1]."""
