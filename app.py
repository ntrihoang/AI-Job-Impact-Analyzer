import sqlite3
from pathlib import Path

import pandas as pd


# ==========================================
# 1. CONFIGURATION
# ==========================================

# Thu muc goc cua project (cung cap voi app.py, data/, outputs/).
# Dat file nay o thu muc goc; chay duoc tu bat ky thu muc lam viec nao.
BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "database" / "career.db"

# Thu muc chua cac CSV ket qua K-Means.
KMEANS_DIR = BASE_DIR / "outputs" / "kmeans_experiments"

# Ten file CSV K=4. Neu de None hoac file khong ton tai,
# chuong trinh tu tim file *k4*.csv trong KMEANS_DIR.
CLUSTER_CSV_NAME = "work_activities_41_lv_k4.csv"

# Ten cot nhan cluster trong CSV (neu khac, sua o day).
CLUSTER_COLUMN = "cluster_id"

TOP_N_ACTIVITIES = 8


# ==========================================
# 2. LOAD DATA
# ==========================================

def find_cluster_csv():
    """Tra ve duong dan CSV clustering K=4."""
    if CLUSTER_CSV_NAME:
        path = KMEANS_DIR / CLUSTER_CSV_NAME
        if path.exists():
            return path

    if not KMEANS_DIR.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc: {KMEANS_DIR}")

    candidates = sorted(KMEANS_DIR.glob("*k4*.csv"))

    if len(candidates) == 1:
        print(f"[Thong bao] Dung file CSV: {candidates[0].name}")
        return candidates[0]

    available = [p.name for p in sorted(KMEANS_DIR.glob("*.csv"))]
    if not candidates:
        raise FileNotFoundError(
            f"Khong co file *k4*.csv trong {KMEANS_DIR}.\n"
            f"Cac CSV hien co: {available}\n"
            "Hay sua CLUSTER_CSV_NAME."
        )

    raise FileNotFoundError(
        f"Co nhieu file K=4: {[p.name for p in candidates]}\n"
        "Hay dat CLUSTER_CSV_NAME la mot trong cac file tren."
    )


def load_data():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay database: {DB_PATH}")

    cluster_csv = find_cluster_csv()

    with sqlite3.connect(DB_PATH) as conn:
        occupations = pd.read_sql_query(
            """
            SELECT occupation_code, occupation_title, ai_exposure_score
            FROM occupations
            """,
            conn,
        )

        activities = pd.read_sql_query(
            """
            SELECT occupation_code, activity,
                   importance_score, level_score
            FROM occupation_work_activities
            """,
            conn,
        )

    clusters = pd.read_csv(cluster_csv)

    if "occupation_code" not in clusters.columns:
        raise ValueError(
            "CSV khong co cot occupation_code. "
            f"Cac cot hien co: {clusters.columns.tolist()}"
        )

    if CLUSTER_COLUMN not in clusters.columns:
        possible = [
            col for col in clusters.columns
            if "cluster_id" in col.lower()
        ]
        raise ValueError(
            f"Khong tim thay cot '{CLUSTER_COLUMN}'. "
            f"Cot co the dung: {possible}. "
            "Hay sua CLUSTER_COLUMN."
        )

    clusters = clusters[["occupation_code", CLUSTER_COLUMN]].copy()
    clusters = clusters.rename(columns={CLUSTER_COLUMN: "cluster"})

    # Dong bo kieu du lieu ma nghe de noi bang.
    for df in (occupations, activities, clusters):
        df["occupation_code"] = df["occupation_code"].astype(str).str.strip()

    # Moi nghe chi co mot nhan cluster.
    if clusters["occupation_code"].duplicated().any():
        raise ValueError(
            "CSV co nhieu dong cho cung occupation_code. "
            "Hay chon CSV ket qua cua mot gia tri K duy nhat."
        )

    return occupations, activities, clusters


# ==========================================
# 3. LOOKUP AN OCCUPATION
# ==========================================

def choose_occupation(keyword, occupations):
    keyword = keyword.strip()

    matches = occupations[
        occupations["occupation_code"].eq(keyword)
        | occupations["occupation_title"].str.contains(
            keyword, case=False, na=False, regex=False
        )
    ]

    if matches.empty:
        print("\nKhong tim thay nghe.")
        return None

    if len(matches) == 1:
        return matches.iloc[0]

    print("\nCac nghe phu hop:")
    for i, (_, row) in enumerate(matches.iterrows(), start=1):
        print(f"{i}. {row['occupation_code']} - {row['title']}")

    choice = input("Chon so thu tu: ").strip()

    if not choice.isdigit() or not (1 <= int(choice) <= len(matches)):
        print("Lua chon khong hop le.")
        return None

    return matches.iloc[int(choice) - 1]


def lookup_occupation(keyword, occupations, activities, clusters):
    occupation = choose_occupation(keyword, occupations)
    if occupation is None:
        return

    code = occupation["occupation_code"]
    title = occupation["occupation_title"]
    exposure = occupation["ai_exposure_score"]

    print("\n" + "=" * 65)
    print(f"NGHE: {title}")
    print(f"Ma nghe: {code}")

    # ----- Work Activities -----
    print("\n[1] WORK ACTIVITIES NOI BAT")

    job_activities = (
        activities[activities["occupation_code"] == code]
        .sort_values(["importance_score", "level_score"], ascending=False)
        .head(TOP_N_ACTIVITIES)
    )

    if job_activities.empty:
        print("Khong co Work Activities.")
    else:
        for _, row in job_activities.iterrows():
            print(
                f"- {row['activity']}"
                f" | Importance: {row['importance_score']:.2f}"
                f" | Level: {row['level_score']:.2f}"
            )

    # ----- Cluster assignment -----
    print("\n[2] CLUSTERING")

    assigned = clusters[clusters["occupation_code"] == code]
    cluster_id = None

    if assigned.empty:
        print("Nghe nay chua co nhan cluster trong CSV.")
    else:
        cluster_id = assigned.iloc[0]["cluster"]
        print(f"Cluster: {cluster_id}")

        cluster_codes = clusters.loc[
            clusters["cluster"] == cluster_id, "occupation_code"
        ]

        profile = (
            activities[activities["occupation_code"].isin(cluster_codes)]
            .groupby("activity")["importance_score"]
            .mean()
            .sort_values(ascending=False)
            .head(TOP_N_ACTIVITIES)
        )

        print("Work Activities co Importance trung binh cao nhat cua cum:")
        for activity, score in profile.items():
            print(f"- {activity}: {score:.2f}")

    # ----- AI Exposure -----
    print("\n[3] AI EXPOSURE")

    if pd.isna(exposure):
        print("Nghe nay khong co AI Exposure score.")
    else:
        print(f"Exposure cua nghe: {exposure:.4f}")

        valid_scores = occupations.dropna(subset=["ai_exposure_score"])
        rank = (valid_scores["ai_exposure_score"] > exposure).sum() + 1
        print(f"Thu hang Exposure: {rank}/{len(valid_scores)} (cao den thap)")

        merged = clusters.merge(
            occupations[["occupation_code", "ai_exposure_score"]],
            on="occupation_code",
            how="left",
        )

        cluster_stats = (
            merged.dropna(subset=["ai_exposure_score"])
            .groupby("cluster")["ai_exposure_score"]
            .agg(["count", "mean", "median"])
            .sort_values("mean", ascending=False)
        )

        print("\nSo sanh voi cac cum (Exposure trung binh):")
        for cid, row in cluster_stats.iterrows():
            marker = " <-- nghe dang tra cuu" if cid == cluster_id else ""
            print(
                f"- Cluster {cid}: mean={row['mean']:.4f}, "
                f"median={row['median']:.4f}, "
                f"so nghe hop le={int(row['count'])}{marker}"
            )

        if cluster_id in cluster_stats.index:
            diff = exposure - cluster_stats.loc[cluster_id, "mean"]
            print(f"\nExposure cua nghe so voi trung binh cum: {diff:+.4f}")

    print("=" * 65)


# ==========================================
# 4. MAIN
# ==========================================

def main():
    occupations, activities, clusters = load_data()

    print("AI JOB TRANSITION MAP - TERMINAL LOOKUP")
    print(f"So nghe trong database: {len(occupations)}")
    print(f"So nghe co nhan cluster: {len(clusters)}")

    while True:
        keyword = input("\nNhap ma nghe/ten nghe (q de thoat): ").strip()

        if keyword.lower() == "q":
            break

        if keyword:
            lookup_occupation(keyword, occupations, activities, clusters)


if __name__ == "__main__":
    main()