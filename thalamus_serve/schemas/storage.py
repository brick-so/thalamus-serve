"""Storage-related schema types for S3 and URLs."""

import re

from pydantic import BaseModel, Field, field_validator


class S3Ref(BaseModel):
    """Reference to an S3 object with bucket and key."""

    bucket: str = Field(..., min_length=3, max_length=63)
    key: str = Field(..., min_length=1)
    region: str | None = None

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$", v):
            raise ValueError("Invalid S3 bucket name")
        return v

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @classmethod
    def from_uri(cls, uri: str) -> "S3Ref":
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}")
        path = uri[5:]
        bucket, key = path.split("/", 1)
        return cls(bucket=bucket, key=key)


class Url(BaseModel):
    """Validated HTTP/HTTPS URL."""

    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must be http:// or https://")
        return v


class S3PresignedUrl(BaseModel):
    """S3 presigned URL with signature validation."""

    url: str

    @field_validator("url")
    @classmethod
    def validate_presigned(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("Presigned URL must be https://")
        if ".s3." not in v and ".s3-" not in v:
            raise ValueError("Not an S3 URL")
        if "X-Amz-Signature" not in v:
            raise ValueError("Missing signature parameter")
        return v
