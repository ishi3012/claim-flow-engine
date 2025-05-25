import json

import pandas as pd

from claimflowengine.clustering.cluster_summary import (
    attach_cluster_labels,
    generate_cluster_labels,
)
from claimflowengine.configs.paths import (
    CLUSTER_LABELS_JSON_PATH,
    CLUSTERING_CLAIMS_LABELED_PATH,
    CLUSTERING_OUTPUT_PATH,
)
from claimflowengine.utils.logger import get_logger

logger = get_logger("cluster")


# Load clustered claims
logger.info("Load clustered claims ...")
df = pd.read_csv(CLUSTERING_OUTPUT_PATH)

# Generate labels
logger.info("Generate labels ...")
labels = generate_cluster_labels(df, method="tfidf", top_n=2)

# Save labels as JSON
logger.info("Save labels as JSON ...")
with open(CLUSTER_LABELS_JSON_PATH, "w") as f:
    json.dump(labels, f, indent=2)

# Attach labels to dataframe
logger.info("Attach labels to dataframe ...")
df_labeled = attach_cluster_labels(df, labels)

# Save updated dataframe
logger.info("Save updated dataframe ...")
df_labeled.to_csv(CLUSTERING_CLAIMS_LABELED_PATH, index=False)

print(f"Labels saved to {CLUSTER_LABELS_JSON_PATH}")
print(f"Labeled claims saved to {CLUSTERING_CLAIMS_LABELED_PATH}")
