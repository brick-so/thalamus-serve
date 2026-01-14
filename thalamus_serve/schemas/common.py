import base64
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class Base64Data(BaseModel):
    data: str
    media_type: str = "application/octet-stream"

    @field_validator("data")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64")
        return v

    def decode(self) -> bytes:
        return base64.b64decode(self.data)


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x2")
    @classmethod
    def x2_gt_x1(cls, v: float, info) -> float:
        if "x1" in info.data and v <= info.data["x1"]:
            raise ValueError("x2 must be > x1")
        return v

    @field_validator("y2")
    @classmethod
    def y2_gt_y1(cls, v: float, info) -> float:
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
    name: str
    confidence: float = Field(..., ge=0, le=1)


class Vector(BaseModel):
    values: list[float] = Field(..., min_length=1)

    @property
    def dim(self) -> int:
        return len(self.values)


class Span(BaseModel):
    text: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    label: str | None = None
    score: float | None = Field(None, ge=0, le=1)


Prob = Annotated[float, Field(ge=0, le=1)]
