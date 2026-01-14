from fastapi.testclient import TestClient

from thalamus_serve import (
    CacheClearResponse,
    HealthResponse,
    PredictResponse,
    ReadyResponse,
    SchemaResponse,
    StatusResponse,
    UnloadResponse,
)
from thalamus_serve.testing import TEST_API_KEY_HEADER


class TestPublicEndpoints:
    def test_health_accessible_without_auth(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_ready_accessible_without_auth(self, client: TestClient) -> None:
        r = client.get("/ready")
        assert r.status_code == 200

    def test_metrics_accessible_without_auth(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_status_accessible_without_auth(self, client: TestClient) -> None:
        r = client.get("/status")
        assert r.status_code == 200


class TestProtectedEndpoints:
    def test_schema_list_requires_auth(self, client: TestClient) -> None:
        r = client.get("/schema")
        assert r.status_code == 401

    def test_schema_by_id_requires_auth(self, client: TestClient) -> None:
        r = client.get("/schema/default")
        assert r.status_code == 401

    def test_predict_requires_auth(self, client: TestClient) -> None:
        r = client.post(
            "/predict", json={"model": "default", "inputs": [{"data": "x"}]}
        )
        assert r.status_code == 401

    def test_cache_clear_requires_auth(self, client: TestClient) -> None:
        r = client.post("/cache/clear")
        assert r.status_code == 401

    def test_unload_requires_auth(self, client: TestClient) -> None:
        r = client.post("/models/default/unload", json={})
        assert r.status_code == 401


class TestHealthContract:
    def test_response_schema(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        data = HealthResponse.model_validate(r.json())
        assert data.status == "ok"

    def test_models_list_not_empty(self, client: TestClient) -> None:
        r = client.get("/health")
        data = HealthResponse.model_validate(r.json())
        assert len(data.models) >= 1

    def test_model_status_fields(self, client: TestClient) -> None:
        r = client.get("/health")
        data = HealthResponse.model_validate(r.json())
        for model in data.models:
            assert isinstance(model.id, str) and model.id
            assert isinstance(model.version, str) and model.version
            assert isinstance(model.ready, bool)
            assert isinstance(model.critical, bool)


class TestReadyContract:
    def test_response_schema(self, client: TestClient) -> None:
        r = client.get("/ready")
        assert r.status_code == 200
        data = ReadyResponse.model_validate(r.json())
        assert isinstance(data.ready, bool)
        assert len(data.models) >= 1

    def test_ready_when_critical_models_ready(self, client: TestClient) -> None:
        r = client.get("/ready")
        data = ReadyResponse.model_validate(r.json())
        assert data.ready is True

    def test_model_includes_critical_flag(self, client: TestClient) -> None:
        r = client.get("/ready")
        data = ReadyResponse.model_validate(r.json())
        for model in data.models:
            assert isinstance(model.critical, bool)


class TestStatusContract:
    def test_response_schema(self, client: TestClient) -> None:
        r = client.get("/status")
        assert r.status_code == 200
        data = StatusResponse.model_validate(r.json())
        assert len(data.models) >= 1

    def test_includes_uptime(self, client: TestClient) -> None:
        r = client.get("/status")
        data = StatusResponse.model_validate(r.json())
        assert data.uptime_seconds >= 0

    def test_includes_cache_info(self, client: TestClient) -> None:
        r = client.get("/status")
        data = StatusResponse.model_validate(r.json())
        assert data.cache is not None
        assert data.cache.max_size_bytes > 0

    def test_includes_gpu_info(self, client: TestClient) -> None:
        r = client.get("/status")
        data = StatusResponse.model_validate(r.json())
        assert data.gpu is not None
        assert "available" in data.gpu


class TestMetricsContract:
    def test_prometheus_format(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")

    def test_requests_metric_exists(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert "thalamus_requests_total" in r.text

    def test_latency_metric_exists(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert "thalamus_latency_seconds" in r.text

    def test_batch_size_metric_exists(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert "thalamus_batch_size" in r.text

    def test_model_info_metric_exists(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert "thalamus_model_info" in r.text

    def test_cache_metrics_exist(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert "thalamus_cache_size_bytes" in r.text
        assert "thalamus_cache_file_count" in r.text

    def test_inference_latency_metric_exists(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert "thalamus_inference_seconds" in r.text


class TestSchemaContract:
    def test_list_response_schema(self, client: TestClient) -> None:
        r = client.get("/schema", headers=TEST_API_KEY_HEADER)
        assert r.status_code == 200
        schemas = [SchemaResponse.model_validate(s) for s in r.json()]
        assert len(schemas) >= 1

    def test_schema_required_fields(self, client: TestClient) -> None:
        r = client.get("/schema", headers=TEST_API_KEY_HEADER)
        schemas = [SchemaResponse.model_validate(s) for s in r.json()]
        for schema in schemas:
            assert isinstance(schema.id, str) and schema.id
            assert isinstance(schema.version, str) and schema.version
            assert isinstance(schema.description, str)
            assert isinstance(schema.input, dict)
            assert isinstance(schema.output, dict)

    def test_by_id_response_schema(self, client: TestClient) -> None:
        r = client.get("/schema/default", headers=TEST_API_KEY_HEADER)
        assert r.status_code == 200
        schema = SchemaResponse.model_validate(r.json())
        assert schema.id == "default"

    def test_not_found(self, client: TestClient) -> None:
        r = client.get("/schema/nonexistent-model", headers=TEST_API_KEY_HEADER)
        assert r.status_code == 404

    def test_versions_endpoint(self, client: TestClient) -> None:
        r = client.get("/schema/default/versions", headers=TEST_API_KEY_HEADER)
        assert r.status_code == 200
        versions = r.json()
        assert isinstance(versions, list)
        assert len(versions) >= 1

    def test_versions_not_found(self, client: TestClient) -> None:
        r = client.get("/schema/nonexistent/versions", headers=TEST_API_KEY_HEADER)
        assert r.status_code == 404


class TestPredictContract:
    def test_response_schema(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"model": "default", "inputs": [{"data": "test"}]},
            headers=TEST_API_KEY_HEADER,
        )
        assert r.status_code == 200
        PredictResponse.model_validate(r.json())

    def test_meta_required_fields(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"model": "default", "inputs": [{"data": "test"}]},
            headers=TEST_API_KEY_HEADER,
        )
        data = PredictResponse.model_validate(r.json())
        assert data.meta.model == "default"
        assert isinstance(data.meta.version, str) and data.meta.version
        assert isinstance(data.meta.latency_ms, float) and data.meta.latency_ms >= 0
        assert data.meta.batch_size == 1

    def test_inference_ms_present(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"model": "default", "inputs": [{"data": "test"}]},
            headers=TEST_API_KEY_HEADER,
        )
        data = PredictResponse.model_validate(r.json())
        assert data.meta.inference_ms is not None
        assert data.meta.inference_ms >= 0

    def test_timing_breakdown_with_hooks(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"model": "default", "inputs": [{"data": "test"}]},
            headers=TEST_API_KEY_HEADER,
        )
        data = PredictResponse.model_validate(r.json())
        assert data.meta.preprocessing_ms is not None
        assert data.meta.preprocessing_ms >= 0
        assert data.meta.postprocessing_ms is not None
        assert data.meta.postprocessing_ms >= 0

    def test_outputs_match_inputs_count(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"model": "default", "inputs": [{"data": "test"}]},
            headers=TEST_API_KEY_HEADER,
        )
        data = PredictResponse.model_validate(r.json())
        assert len(data.outputs) == 1

    def test_batch_processing(self, client: TestClient) -> None:
        inputs = [{"data": f"item-{i}"} for i in range(3)]
        r = client.post(
            "/predict",
            json={"model": "default", "inputs": inputs},
            headers=TEST_API_KEY_HEADER,
        )
        assert r.status_code == 200
        data = PredictResponse.model_validate(r.json())
        assert len(data.outputs) == 3
        assert data.meta.batch_size == 3

    def test_model_not_found(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"model": "nonexistent-model", "inputs": [{"data": "test"}]},
            headers=TEST_API_KEY_HEADER,
        )
        assert r.status_code == 404

    def test_empty_inputs_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"model": "default", "inputs": []},
            headers=TEST_API_KEY_HEADER,
        )
        assert r.status_code == 422

    def test_missing_inputs_field_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"model": "default"},
            headers=TEST_API_KEY_HEADER,
        )
        assert r.status_code == 422


class TestdefaultModel:
    def test_predict_without_model_uses_default(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"inputs": [{"data": "test"}]},
            headers=TEST_API_KEY_HEADER,
        )
        assert r.status_code == 200
        data = PredictResponse.model_validate(r.json())
        assert data.meta.model == "default"


class TestCacheManagement:
    def test_clear_response_schema(self, client: TestClient) -> None:
        r = client.post("/cache/clear", headers=TEST_API_KEY_HEADER)
        assert r.status_code == 200
        data = CacheClearResponse.model_validate(r.json())
        assert data.cleared_bytes >= 0
        assert data.cleared_files >= 0


class TestModelUnload:
    def test_unload_unknown_model_404(self, client: TestClient) -> None:
        r = client.post(
            "/models/nonexistent/unload",
            json={},
            headers=TEST_API_KEY_HEADER,
        )
        assert r.status_code == 404

    def test_unload_response_schema(self, client: TestClient) -> None:
        r = client.post(
            "/models/default/unload",
            json={},
            headers=TEST_API_KEY_HEADER,
        )
        assert r.status_code == 200
        data = UnloadResponse.model_validate(r.json())
        assert data.model_id == "default"
        assert isinstance(data.versions_unloaded, list)
