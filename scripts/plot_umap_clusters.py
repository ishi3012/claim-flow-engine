import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from claimflowengine.configs.paths import CLUSTERING_OUTPUT_PATH


def plot_umap_clusters(input_csv: str, output_path: str) -> None:
    df = pd.read_csv(input_csv)

    if "umap_x" not in df.columns or "umap_y" not in df.columns:
        raise ValueError("Missing `umap_x` and `umap_y` columns in input file.")

    if "denial_cluster_id" not in df.columns:
        raise ValueError("Missing `denial_cluster_id` column in input file.")

    # Normalize cluster labels
    le = LabelEncoder()
    df["cluster_label"] = le.fit_transform(df["denial_cluster_id"].astype(str))

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="umap_x",
        y="umap_y",
        hue="cluster_label",
        palette="tab10",
        data=df,
        legend="full",
        alpha=0.7,
    )
    plt.title("UMAP Projection of Denial Clusters", fontsize=16)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved UMAP cluster plot to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=CLUSTERING_OUTPUT_PATH, help="CSV with UMAP results"
    )
    parser.add_argument(
        "--output", default="data/umap_cluster_plot.png", help="Output image path"
    )
    args = parser.parse_args()

    plot_umap_clusters(args.input, args.output)
