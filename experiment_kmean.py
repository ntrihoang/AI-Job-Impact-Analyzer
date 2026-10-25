import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. CONFIGURATION
# =========================================================

DB_PATH = Path("data/database/career.db")
OUTPUT_DIR = Path("outputs/kmeans_experiments")

RANDOM_STATE = 42

# Thử K từ 2 đến 8
K_VALUES = range(2, 9)


# =========================================================
# 2. LOAD WORK ACTIVITIES DIRECTLY FROM SQLITE
# =========================================================

def load_feature_matrix(db_path):
    with sqlite3.connect(db_path) as conn:
        work_activities = pd.read_sql_query(
            """
            SELECT
                occupation_code,
                activity,
                level_score
            FROM occupation_work_activities
            """,
            conn,
        )

    required_columns = {
        "occupation_code",
        "activity",
        "level_score",
    }

    missing_columns = required_columns - set(work_activities.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if work_activities["occupation_code"].isna().any():
        raise ValueError("Found NULL occupation_code.")

    if work_activities["activity"].isna().any():
        raise ValueError("Found NULL activity.")

    if work_activities.duplicated(
        ["occupation_code", "activity"]
    ).any():
        raise ValueError(
            "Duplicate occupation-activity pairs detected."
        )

    # Chuyển dữ liệu dài sang ma trận 41 LV
    X_41 = work_activities.pivot(
        index="occupation_code",
        columns="activity",
        values="level_score",
    )

    X_41.columns.name = None
    X_41.index.name = "occupation_code"

    if X_41.shape[1] != 41:
        raise ValueError(
            f"Expected 41 activities, got {X_41.shape[1]}."
        )

    missing_rows = X_41.isna().any(axis=1)

    print("=== WORK ACTIVITIES DATA ===")
    print("Total occupations:", len(X_41))
    print("Total activities:", X_41.shape[1])
    print(
        "Occupations with missing LV:",
        int(missing_rows.sum()),
    )

    # Chỉ giữ nghề đủ 41 LV
    X_41 = X_41.loc[~missing_rows].copy()

    if not np.isfinite(X_41.to_numpy()).all():
        raise ValueError(
            "Non-finite values found in 41 LV features."
        )

    if len(X_41) < max(K_VALUES) + 1:
        raise ValueError(
            "Too few occupations for the requested K values."
        )

    print("Valid occupations:", len(X_41))
    print("41 LV matrix shape:", X_41.shape)

    return X_41


# =========================================================
# 3. PREPROCESSING
# =========================================================

def prepare_data(X_41):
    scaler = StandardScaler()

    X_41_scaled = scaler.fit_transform(X_41)

    print("\n=== DATA STANDARDIZATION ===")
    print("Features:", X_41.shape[1])
    print("Shape:", X_41_scaled.shape)

    return X_41.index, X_41_scaled


# =========================================================
# 4. EVALUATE CLUSTERING
# =========================================================

def evaluate_clustering(X, labels):
    number_of_clusters = len(np.unique(labels))

    if (
        number_of_clusters < 2
        or number_of_clusters >= len(labels)
    ):
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }


# =========================================================
# 5. RUN K-MEANS EXPERIMENTS
# =========================================================

def run_experiments(
    occupation_codes,
    X,
):
    results = []
    assignments = {}

    print("\n=== K-MEANS ON 41 LV FEATURES ===")

    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=RANDOM_STATE,
        )

        labels = model.fit_predict(X)

        metrics = evaluate_clustering(
            X,
            labels,
        )

        cluster_sizes = (
            pd.Series(labels)
            .value_counts()
            .sort_index()
        )

        result = {
            "k": k,
            "inertia": model.inertia_,
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics[
                "calinski_harabasz"
            ],
            "smallest_cluster": int(
                cluster_sizes.min()
            ),
            "largest_cluster": int(
                cluster_sizes.max()
            ),
            "n_clusters_found": int(
                len(cluster_sizes)
            ),
        }

        results.append(result)

        assignments[k] = pd.DataFrame(
            {
                "occupation_code": occupation_codes.to_numpy(),
                "cluster_id": labels,
            }
        )

        print(
            f"K={k} | "
            f"Silhouette={metrics['silhouette']:.4f} | "
            f"DB={metrics['davies_bouldin']:.4f} | "
            f"CH={metrics['calinski_harabasz']:.2f} | "
            f"Inertia={model.inertia_:.2f} | "
            f"Cluster sizes={sorted(cluster_sizes.tolist())}"
        )

    return pd.DataFrame(results), assignments


# =========================================================
# 6. SAVE RESULTS
# =========================================================

def save_results(
    results,
    assignments,
    output_dir,
):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = results.sort_values("k")

    results.to_csv(
        output_dir / "kmeans_comparison.csv",
        index=False,
    )

    for k, assignment in assignments.items():
        assignment.to_csv(
            output_dir / f"work_activities_41_lv_k{k}.csv",
            index=False,
        )

    print(
        "\nResults saved to:",
        output_dir.resolve(),
    )


# =========================================================
# 7. MAIN
# =========================================================

def main():
    X_41 = load_feature_matrix(DB_PATH)

    occupation_codes, X_scaled = prepare_data(X_41)

    results, assignments = run_experiments(
        occupation_codes,
        X_scaled,
    )

    save_results(
        results,
        assignments,
        OUTPUT_DIR,
    )

    print("\n=== TOP RESULTS BY SILHOUETTE ===")

    print(
        results.sort_values(
            "silhouette",
            ascending=False,
        )[
            [
                "k",
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "smallest_cluster",
                "largest_cluster",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()