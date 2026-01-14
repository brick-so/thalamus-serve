"""Weight source configuration types for model loading."""

from pydantic import BaseModel, Field, model_validator


class S3Weight(BaseModel, frozen=True):
    """S3 weight source configuration.

    Supports both single file downloads (key) and directory downloads (prefix)
    for sharded models.

    Args:
        bucket: S3 bucket name.
        key: Object key path for single file download.
        prefix: S3 prefix for directory download (all objects under this prefix).
        region: Optional AWS region (uses default if not specified).

    Examples:
        Single file: S3Weight(bucket="models", key="bert/model.pt")
        Sharded: S3Weight(bucket="models", prefix="llama/weights/")
    """

    type: str = Field("s3", init=False)
    bucket: str
    key: str | None = None
    prefix: str | None = None
    region: str | None = None

    @model_validator(mode="after")
    def validate_key_or_prefix(self) -> "S3Weight":
        if self.key is None and self.prefix is None:
            raise ValueError("Either 'key' or 'prefix' must be specified")
        if self.key is not None and self.prefix is not None:
            raise ValueError("Cannot specify both 'key' and 'prefix'")
        return self

    @property
    def is_directory(self) -> bool:
        """True if this downloads a directory (prefix), False for single file."""
        return self.prefix is not None


class HFWeight(BaseModel, frozen=True):
    """HuggingFace Hub weight source configuration.

    Args:
        repo: Repository ID (e.g., "bert-base-uncased").
        filename: Specific file to download, or None for full repo snapshot.
        revision: Git revision (branch, tag, or commit hash).
    """

    type: str = Field("hf", init=False)
    repo: str
    filename: str | None = None
    revision: str = "main"

    @property
    def is_snapshot(self) -> bool:
        """True if this downloads entire repo (directory), False for single file."""
        return self.filename is None


class HTTPWeight(BaseModel, frozen=True):
    """HTTP/HTTPS weight source configuration.

    Args:
        urls: List of URLs to download. Single URL returns a file path,
              multiple URLs returns a directory path.

    Examples:
        Single file: HTTPWeight(urls=["https://example.com/model.pt"])
        Sharded: HTTPWeight(urls=[
            "https://example.com/model-00001.pt",
            "https://example.com/model-00002.pt",
        ])
    """

    type: str = Field("http", init=False)
    urls: tuple[str, ...]

    @model_validator(mode="after")
    def validate_urls(self) -> "HTTPWeight":
        if len(self.urls) == 0:
            raise ValueError("'urls' must not be empty")
        return self

    @property
    def is_directory(self) -> bool:
        """True if this downloads multiple files (directory), False for single file."""
        return len(self.urls) > 1


WeightSource = S3Weight | HFWeight | HTTPWeight
"""Union type for all supported weight sources."""
