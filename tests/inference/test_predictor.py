from mock_claim import mock_input

from claimflowengine.inference.loader import load_model
from claimflowengine.inference.predictor import predict_claims


def test_predict_claims_output_format() -> None:
    raw = mock_input
    model, target_encoder, numeric_transformer = load_model()

    results = predict_claims(raw, model, target_encoder, numeric_transformer)

    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)
    for r in results:
        assert "denied" in r
        assert "denial_probability" in r
