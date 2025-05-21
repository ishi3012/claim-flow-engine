from claimflowengine.inference.loader import load_model


def test_load_model_success() -> None:
    model, target_encoder, numeric_transformer = load_model()
    assert model is not None
    assert target_encoder is not None
    assert numeric_transformer is not None
