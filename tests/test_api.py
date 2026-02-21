"""
Tests for the Infant Growth Digital Twin API v3.0
Run: pytest tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient

# These tests work against the running API or via TestClient
# For TestClient, models must be pre-trained

try:
    from src.api.server import app
    client = TestClient(app)
    HAS_APP = True
except Exception:
    HAS_APP = False


@pytest.mark.skipif(not HAS_APP, reason="App not loadable (models not trained?)")
class TestHealthAndInfo:

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["version"] == "3.0.0"

    def test_model_info(self):
        r = client.get("/model/info")
        assert r.status_code == 200
        data = r.json()
        assert "metadata" in data
        assert "comparison_table" in data

    def test_docs_available(self):
        r = client.get("/docs")
        assert r.status_code == 200


@pytest.mark.skipif(not HAS_APP, reason="App not loadable")
class TestInfantCRUD:

    def test_create_infant(self):
        r = client.post("/infants", json={
            "infant_id": "test-001",
            "sex": "female",
            "name": "Test Baby",
            "birth_weight_kg": 3.2,
        })
        assert r.status_code in (201, 409)  # 409 if already exists

    def test_list_infants(self):
        r = client.get("/infants")
        assert r.status_code == 200
        assert "count" in r.json()

    def test_get_infant(self):
        # Ensure infant exists
        client.post("/infants", json={
            "infant_id": "test-002", "sex": "male"
        })
        r = client.get("/infants/test-002")
        assert r.status_code == 200

    def test_get_nonexistent(self):
        r = client.get("/infants/nonexistent-999")
        assert r.status_code == 404


@pytest.mark.skipif(not HAS_APP, reason="App not loadable")
class TestObservationsAndPredictions:

    @pytest.fixture(autouse=True)
    def setup_infant(self):
        client.post("/infants", json={
            "infant_id": "test-predict",
            "sex": "male",
            "birth_weight_kg": 3.5,
        })

    def test_add_observation(self):
        r = client.post("/infants/test-predict/observations", json={
            "age_months": 3.0,
            "metric": "weight_for_age",
            "value": 6.2,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["z_score"] is not None
        assert data["percentile"] is not None

    def test_get_zscores(self):
        # Add observation first
        client.post("/infants/test-predict/observations", json={
            "age_months": 3.0,
            "metric": "weight_for_age",
            "value": 6.2,
        })
        r = client.get("/infants/test-predict/zscore")
        assert r.status_code == 200
        assert len(r.json()) > 0

    def test_predict_growth(self):
        # Add observation
        client.post("/infants/test-predict/observations", json={
            "age_months": 3.0,
            "metric": "weight_for_age",
            "value": 6.2,
        })
        r = client.get("/infants/test-predict/predict",
                       params={"metric": "weight_for_age", "months": "6,9,12"})
        assert r.status_code == 200
        preds = r.json()
        assert len(preds) == 3
        for p in preds:
            assert "predicted_value" in p
            assert "ci_lower" in p
            assert "ci_upper" in p
            assert p["ci_lower"] <= p["predicted_value"] <= p["ci_upper"]

    def test_alerts(self):
        r = client.get("/infants/test-predict/alerts")
        assert r.status_code == 200


@pytest.mark.skipif(not HAS_APP, reason="App not loadable")
class TestWHOEndpoints:

    def test_percentile_lines(self):
        r = client.get("/who/percentile-lines",
                       params={"metric": "weight_for_age", "sex": "male"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["lines"]) == 5  # P3, P15, P50, P85, P97

    def test_percentile_lines_female(self):
        r = client.get("/who/percentile-lines",
                       params={"metric": "length_for_age", "sex": "female"})
        assert r.status_code == 200


@pytest.mark.skipif(not HAS_APP, reason="App not loadable")
class TestValidation:

    def test_invalid_sex(self):
        r = client.post("/infants", json={
            "infant_id": "bad-sex", "sex": "unknown"
        })
        assert r.status_code == 422

    def test_invalid_metric(self):
        client.post("/infants", json={
            "infant_id": "test-val", "sex": "female"
        })
        r = client.post("/infants/test-val/observations", json={
            "age_months": 3.0, "metric": "invalid_metric", "value": 5.0
        })
        assert r.status_code == 422

    def test_negative_weight(self):
        client.post("/infants", json={
            "infant_id": "test-neg", "sex": "male"
        })
        r = client.post("/infants/test-neg/observations", json={
            "age_months": 3.0, "metric": "weight_for_age", "value": -1.0
        })
        assert r.status_code == 422
