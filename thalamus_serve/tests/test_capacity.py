from thalamus_serve import CapacityResponse, ModelCapacity


class TestCapacitySchemas:
    def test_model_capacity_defaults(self) -> None:
        cap = ModelCapacity(remaining_requests=2, ideal_batch_size=8, max_batch_size=16)
        assert cap.accepting is True
        assert cap.reason is None

    def test_capacity_response_shape(self) -> None:
        response = CapacityResponse(
            accepting=True,
            remaining_requests=2,
            models={
                "a@1.0.0": ModelCapacity(
                    remaining_requests=2, ideal_batch_size=8, max_batch_size=16
                )
            },
            inflight_requests=0,
            uptime_seconds=1.5,
        )
        assert response.models["a@1.0.0"].max_batch_size == 16
