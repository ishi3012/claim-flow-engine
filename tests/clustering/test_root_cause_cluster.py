from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# from numpy.typing import NDArray
from claimflowengine.clustering.root_cause_cluster import run_clustering_pipeline
from tests.clustering.mock_cluster import mock_cluster_input


@pytest.fixture  # type: ignore[misc]
def mocked_embeddings() -> np.ndarray[Any, Any]:
    return np.random.rand(mock_cluster_input.shape[0], 384)


@patch("claimflowengine.clustering.root_cause_cluster.embed_denial_reasons")
def test_run_clustering_pipeline(
    mock_embed_fn: MagicMock, mocked_embeddings: np.ndarray[Any, Any]
) -> None:
    mock_embed_fn.return_value = mocked_embeddings

    result = run_clustering_pipeline(
        df=mock_cluster_input,
        text_col="denial_reason",
        notes_col="followup_notes",
        use_notes=True,
        id_col="claim_id",
        model_name="mock-model",
        output_path="dummy.csv",
        cluster_model_path="dummy_cluster.pkl",
        reducer_model_path="dummy_reducer.pkl",
    )

    assert isinstance(result, pd.DataFrame)
    assert "denial_cluster_id" in result.columns
    assert len(result) == len(mock_cluster_input)
