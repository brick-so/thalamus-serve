import json
import os
from pathlib import Path

from pydantic import BaseModel, Field


class S3Weight(BaseModel, frozen=True):
    type: str = Field("s3", init=False)
    bucket: str
    key: str
    region: str | None = None


class HFWeight(BaseModel, frozen=True):
    type: str = Field("hf", init=False)
    repo: str
    filename: str | None = None
    revision: str = "main"


class HTTPWeight(BaseModel, frozen=True):
    type: str = Field("http", init=False)
    url: str


WeightSource = S3Weight | HFWeight | HTTPWeight


class ModelConfig(BaseModel, frozen=True):
    weights: dict[str, WeightSource] = Field(default_factory=dict)
    device: str = "auto"


class DeployConfig(BaseModel, frozen=True):
    name: str = ""
    health_check_timeout: int = 600
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    secrets_enabled: bool = False
    secrets_parameters: list[str] = Field(default_factory=list)


_config: DeployConfig | None = None


def _parse_weight(data: dict) -> WeightSource | None:
    weight_type = data.get("type")

    if weight_type == "s3":
        return S3Weight(
            bucket=data["bucket"],
            key=data["key"],
            region=data.get("region"),
        )
    if weight_type == "hf":
        return HFWeight(
            repo=data["repo"],
            filename=data.get("filename"),
            revision=data.get("revision", "main"),
        )
    if weight_type == "http":
        return HTTPWeight(url=data["url"])

    return None


def _load_deploy_json() -> dict:
    config_path = os.environ.get("THALAMUS_DEPLOY_CONFIG")
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).parent.parent / "thalamus-deploy.json"

    if not path.exists():
        return {}

    with open(path) as f:
        return json.load(f)


def _get_required_secrets(models: dict[str, ModelConfig]) -> set[str]:
    required: set[str] = set()

    for model_config in models.values():
        for weight in model_config.weights.values():
            if isinstance(weight, S3Weight):
                required.add("AWS_ACCESS_KEY_ID")
                required.add("AWS_SECRET_ACCESS_KEY")
            elif isinstance(weight, HFWeight):
                required.add("HF_TOKEN")

    return required


def get_config() -> DeployConfig:
    global _config
    if _config is not None:
        return _config

    data = _load_deploy_json()
    models_data = data.get("models", {})
    secrets_data = data.get("secrets", {})

    models: dict[str, ModelConfig] = {}
    for model_key, cfg in models_data.items():
        weights_data = cfg.get("weights", {})
        weights: dict[str, WeightSource] = {}
        for weight_name, weight_cfg in weights_data.items():
            parsed = _parse_weight(weight_cfg)
            if parsed:
                weights[weight_name] = parsed

        models[model_key] = ModelConfig(
            weights=weights,
            device=cfg.get("device", "auto"),
        )

    _config = DeployConfig(
        name=data.get("name", ""),
        health_check_timeout=data.get("healthCheckTimeout", 600),
        models=models,
        secrets_enabled=secrets_data.get("enabled", False),
        secrets_parameters=secrets_data.get("parameters", []),
    )
    return _config


def get_model_config(model_id: str, version: str | None = None) -> ModelConfig:
    config = get_config()

    if version:
        versioned_key = f"{model_id}@{version}"
        if versioned_key in config.models:
            return config.models[versioned_key]

    if model_id in config.models:
        return config.models[model_id]

    return ModelConfig()


def validate_secrets() -> list[str]:
    config = get_config()
    required = _get_required_secrets(config.models)
    configured = set(config.secrets_parameters)
    missing = required - configured
    return list(missing)


def reset_config() -> None:
    global _config
    _config = None
