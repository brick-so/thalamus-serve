"""Model specification and registry for managing registered models."""

from typing import Any, get_args, get_origin, get_type_hints

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
    def _infer_types_from_predict(
        cls, model_cls: type
    ) -> tuple[type[BaseModel], type[BaseModel]]:
        hints = get_type_hints(model_cls.predict)
        params = [k for k in hints if k != "return"]
        if not params:
            raise TypeError(f"{model_cls.__name__}.predict needs typed params")

        input_hint = hints[params[0]]
        input_origin = get_origin(input_hint)
        input_args = get_args(input_hint)
        input_type = (
            input_args[0] if input_origin is list and input_args else input_hint
        )

        output_hint = hints.get("return")
        output_origin = get_origin(output_hint)
        output_args = get_args(output_hint)
        output_type = (
            output_args[0] if output_origin is list and output_args else output_hint
        )

        return input_type, output_type

    @classmethod
    def from_class(
        cls,
        model_cls: type,
        model_id: str | None = None,
        version: str = "1.0.0",
        description: str | None = None,
        default: bool = False,
        default_version: bool = False,
        critical: bool = True,
        weights: dict[str, WeightSource] | None = None,
        device: str = "auto",
        input_type: type[BaseModel] | None = None,
        output_type: type[BaseModel] | None = None,
    ) -> "ModelSpec":
        mid = model_id or model_cls.__name__
        desc = description or model_cls.__doc__ or ""

        if input_type and output_type:
            resolved_input = input_type
            resolved_output = output_type
        else:
            resolved_input, resolved_output = cls._infer_types_from_predict(model_cls)

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
            input_type=resolved_input,
            output_type=resolved_output,
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
