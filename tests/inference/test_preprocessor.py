import pandas as pd
from mock_claim import mock_input

from claimflowengine.inference.loader import load_model
from claimflowengine.inference.preprocessor import preprocess_for_inference


def test_preprocess_mock_claim() -> None:
    model, target_encoder, numeric_transformer = load_model()

    features = preprocess_for_inference(
        raw_input=mock_input,
        target_encoder=target_encoder,
        numeric_transformer=numeric_transformer,
    )
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == 1
    assert not features.isnull().any().any()
    assert features.shape[1] > 0
