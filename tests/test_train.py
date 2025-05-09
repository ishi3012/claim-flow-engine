def test_train_model_runs():
    from claimflowengine.pipelines.train import train_model
    import yaml
    from claimflowengine.utils.paths import CONFIG_PATH, MODEL_DIR

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    model, metrics = train_model(config)

    assert metrics["auc"] > 0.5
    assert "f1" in metrics

    # Confirm model.pkl saved
    model_path = MODEL_DIR / "model.pkl"
    assert model_path.exists()
