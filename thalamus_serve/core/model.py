"""Model specification and registry for managing registered models."""

from typing import Any

from packaging.version import Version
from pydantic import BaseModel

from thalamus_serve.config import WeightSource
from thalamus_serve.infra.gpu import GPUAllocator


class ModelSpec:
    """Specification for a registered model including metadata and configuration."""

    def __init__(
        self,
        model_id: str,
        version: str,
        description: str,
        cls: type,
        input_type: type[BaseModel],
        output_type: type[BaseModel],
        has_preprocess: bool = False,
        has_postprocess: bool = False,
        is_default: bool = False,
        is_default_version: bool = False,
        is_critical: bool = True,
        weights: dict[str, WeightSource] | None = None,
        device_preference: str = "auto",
    ) -> None:
        self.id = model_id
        self.version = version
        self.description = description
        self.cls = cls
        self.input_type = input_type
        self.output_type = output_type
        self.has_preprocess = has_preprocess
        self.has_postprocess = has_postprocess
        self.is_default = is_default
        self.is_default_version = is_default_version
        self.is_critical = is_critical
        self.weights = weights or {}
        self.device_preference = device_preference
        self.instance: Any = None
        self.device: str | None = None

    @classmethod
    def from_class(
        cls,
        model_cls: type,
        model_id: str | None,
        version: str,
        description: str | None,
        default: bool,
        default_version: bool,
        critical: bool,
        weights: dict[str, WeightSource] | None,
        device: str,
        input_type: type[BaseModel],
        output_type: type[BaseModel],
    ) -> "ModelSpec":
        """Create a ModelSpec from a model class.

        Args:
            model_cls: The model class to create a spec for.
            model_id: Unique identifier. Defaults to class name if None.
            version: Semantic version string.
            description: Human-readable description. Defaults to docstring if None.
            default: Whether this is the default model.
            default_version: Whether this is the default version.
            critical: Whether this model is critical for readiness.
            weights: Weight source configurations.
            device: Device preference.
            input_type: Pydantic model for input validation (required).
            output_type: Pydantic model for output serialization (required).

        Returns:
            A configured ModelSpec instance.
        """
        mid = model_id or model_cls.__name__
        desc = description or model_cls.__doc__ or ""

        has_preprocess = hasattr(model_cls, "preprocess") and callable(
            getattr(model_cls, "preprocess", None)
        )
        has_postprocess = hasattr(model_cls, "postprocess") and callable(
            getattr(model_cls, "postprocess", None)
        )

        return cls(
            model_id=mid,
            version=version,
            description=desc.strip(),
            cls=model_cls,
            input_type=input_type,
            output_type=output_type,
            has_preprocess=has_preprocess,
            has_postprocess=has_postprocess,
            is_default=default,
            is_default_version=default_version,
            is_critical=critical,
            weights=weights,
            device_preference=device,
        )


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, dict[str, ModelSpec]] = {}
        self._default_model: str | None = None
        self._default_versions: dict[str, str] = {}

    def register(self, spec: ModelSpec) -> None:
        if spec.id not in self._models:
            self._models[spec.id] = {}
        self._models[spec.id][spec.version] = spec

        if spec.is_default:
            self._default_model = spec.id

        if spec.is_default_version:
            self._default_versions[spec.id] = spec.version

    def get(self, model_id: str, version: str | None = None) -> ModelSpec | None:
        model_versions = self._models.get(model_id)
        if not model_versions:
            return None

        if version is None or version == "latest":
            version = self._resolve_default_version(model_id)

        return model_versions.get(version)

    def get_default(self) -> ModelSpec | None:
        if not self._default_model:
            return None
        return self.get(self._default_model)

    def get_versions(self, model_id: str) -> list[str]:
        model_versions = self._models.get(model_id)
        if not model_versions:
            return []
        return sorted(model_versions.keys(), key=Version, reverse=True)

    def all(self) -> list[ModelSpec]:
        result: list[ModelSpec] = []
        for versions in self._models.values():
            result.extend(versions.values())
        return result

    def all_for_model(self, model_id: str) -> list[ModelSpec]:
        model_versions = self._models.get(model_id)
        if not model_versions:
            return []
        return list(model_versions.values())

    def is_loaded(self, model_id: str, version: str | None = None) -> bool:
        spec = self.get(model_id, version)
        return spec is not None and spec.instance is not None

    def unload(self, model_id: str, version: str | None = None) -> list[str]:
        unloaded: list[str] = []

        def _unload_spec(spec: ModelSpec) -> None:
            if hasattr(spec.instance, "unload"):
                spec.instance.unload()
            if spec.device:
                GPUAllocator.get().release(spec.device)
                spec.device = None
            spec.instance = None

        if version is not None:
            spec = self.get(model_id, version)
            if spec and spec.instance is not None:
                _unload_spec(spec)
                unloaded.append(version)
        else:
            for spec in self.all_for_model(model_id):
                if spec.instance is not None:
                    _unload_spec(spec)
                    unloaded.append(spec.version)

        return unloaded

    def _resolve_default_version(self, model_id: str) -> str:
        if model_id in self._default_versions:
            return self._default_versions[model_id]

        versions = self.get_versions(model_id)
        return versions[0] if versions else ""
