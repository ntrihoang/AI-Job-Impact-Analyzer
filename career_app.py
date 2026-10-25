#!/usr/bin/env python3
"""AI Job Transition Map - ứng dụng tra cứu nghề trên terminal.

Ghép 2 phần:
  - Hồ sơ nghề, Work Activities, Clustering, AI Exposure (SQLite + CSV K-Means)
  - Career Transition và Skill gap (O*NET skills LV, cosine similarity)

Chạy:
  python career_app.py
  python career_app.py --code 15-1252.00
  python career_app.py --code 15-1252.00 --target 15-1243.00 --top 5
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:  # Tránh lỗi font tiếng Việt trên terminal Windows.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "database" / "career.db"
KMEANS_DIR = BASE_DIR / "outputs" / "kmeans_experiments"
CLUSTER_CSV_NAME = "work_activities_41_lv_k4.csv"
CLUSTER_COLUMN = "cluster_id"

ONET_DIR = BASE_DIR / "data" / "raw" / "onet"

TOP_N_ACTIVITIES = 8      # số Work Activities hiển thị
TOP_N_TRANSITION = 5      # số nghề đích gợi ý
TOP_N_SKILLS = 8          # số kỹ năng mỗi nhóm gap
ZERO_GAP_TOL = 0.25       # |gap| <= ngưỡng này coi là "bằng 0" (thang LV 0-7)
WIDTH = 78


# ==========================================
# 2. TIỆN ÍCH IN
# ==========================================

def banner(text):
    print("\n" + "=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def section(text):
    print("\n" + "-" * WIDTH)
    print(text)
    print("-" * WIDTH)


def short(text, n=48):
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


# ==========================================
# 3. LOAD DỮ LIỆU: DATABASE + CLUSTER
# ==========================================

def find_cluster_csv():
    if CLUSTER_CSV_NAME:
        path = KMEANS_DIR / CLUSTER_CSV_NAME
        if path.exists():
            return path

    if not KMEANS_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {KMEANS_DIR}")

    candidates = sorted(KMEANS_DIR.glob("*k4*.csv"))
    if len(candidates) == 1:
        print(f"[Thông báo] Dùng file CSV: {candidates[0].name}")
        return candidates[0]

    available = [p.name for p in sorted(KMEANS_DIR.glob("*.csv"))]
    if not candidates:
        raise FileNotFoundError(
            f"Không có file *k4*.csv trong {KMEANS_DIR}.\n"
            f"Các CSV hiện có: {available}\nHãy sửa CLUSTER_CSV_NAME."
        )
    raise FileNotFoundError(
        f"Có nhiều file K=4: {[p.name for p in candidates]}\n"
        "Hãy đặt CLUSTER_CSV_NAME là một trong các file trên."
    )


def load_db_data():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy database: {DB_PATH}")

    cluster_csv = find_cluster_csv()

    with sqlite3.connect(DB_PATH) as conn:
        occupations = pd.read_sql_query(
            "SELECT occupation_code, occupation_title, ai_exposure_score "
            "FROM occupations",
            conn,
        )
        activities = pd.read_sql_query(
            "SELECT occupation_code, activity, importance_score, level_score "
            "FROM occupation_work_activities",
            conn,
        )

    clusters = pd.read_csv(cluster_csv)

    if "occupation_code" not in clusters.columns:
        raise ValueError(
            "CSV không có cột occupation_code. "
            f"Các cột hiện có: {clusters.columns.tolist()}"
        )
    if CLUSTER_COLUMN not in clusters.columns:
        possible = [c for c in clusters.columns if "cluster" in c.lower()]
        raise ValueError(
            f"Không tìm thấy cột '{CLUSTER_COLUMN}'. Cột có thể dùng: {possible}."
        )

    clusters = clusters[["occupation_code", CLUSTER_COLUMN]].rename(
        columns={CLUSTER_COLUMN: "cluster"}
    )

    for df in (occupations, activities, clusters):
        df["occupation_code"] = df["occupation_code"].astype(str).str.strip()

    if clusters["occupation_code"].duplicated().any():
        raise ValueError(
            "CSV có nhiều dòng cho cùng occupation_code. "
            "Hãy chọn CSV của một giá trị K duy nhất."
        )

    return occupations, activities, clusters


# ==========================================
# 4. LOAD DỮ LIỆU: O*NET SKILLS (LV)
# ==========================================

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\nHãy đặt các CSV trong {ONET_DIR}."
        )
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, low_memory=False)


def load_skill_data():
    essential = read_csv(ONET_DIR / "essential_skills.csv")
    transferable = read_csv(ONET_DIR / "transferable_skills.csv")
    for df in (essential, transferable):
        df.columns = [str(c).strip() for c in df.columns]

    needed = {"O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Data Value"}
    for name, df in [("essential_skills.csv", essential),
                     ("transferable_skills.csv", transferable)]:
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{name} thiếu cột {sorted(missing)}")

    skills = pd.concat([essential, transferable], ignore_index=True)
    for col in ("O*NET-SOC Code", "Element ID", "Scale ID"):
        skills[col] = skills[col].astype(str).str.strip()
    skills["Data Value"] = pd.to_numeric(skills["Data Value"], errors="coerce")
    skills = skills[(skills["Scale ID"] == "LV") & skills["Data Value"].notna()].copy()

    skills = skills.groupby(
        ["O*NET-SOC Code", "Element ID", "Element Name"], as_index=False
    )["Data Value"].mean()

    matrix = skills.pivot_table(
        index="O*NET-SOC Code", columns="Element ID",
        values="Data Value", aggfunc="mean",
    ).sort_index()
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.mean()).fillna(0.0)

    stds = matrix.std(axis=0, ddof=0).replace(0, 1.0)
    z_matrix = (matrix - matrix.mean(axis=0)) / stds

    skill_names = (
        skills.drop_duplicates("Element ID")
        .set_index("Element ID")["Element Name"].to_dict()
    )
    return matrix, z_matrix, skill_names


def resolve_code(code, index):
    """Khớp mã nghề giữa DB và O*NET (vd '15-1252' <-> '15-1252.00')."""
    code = str(code).strip()
    for candidate in (code, f"{code}.00", f"{code.split('.')[0]}.00"):
        if candidate in index:
            return candidate
    return None


# ==========================================
# 5. TÌM NGHỀ
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
        print("\nKhông tìm thấy nghề.")
        return None
    if len(matches) == 1:
        return matches.iloc[0]

    print("\nCác nghề phù hợp:")
    shown = matches.head(30)
    for i, (_, row) in enumerate(shown.iterrows(), start=1):
        print(f"{i}. {row['occupation_code']} - {row['occupation_title']}")
    if len(matches) > len(shown):
        print(f"... và {len(matches) - len(shown)} nghề khác, hãy nhập từ khóa cụ thể hơn.")

    choice = input("Chọn số thứ tự: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(shown)):
        print("Lựa chọn không hợp lệ.")
        return None
    return shown.iloc[int(choice) - 1]


# ==========================================
# 6. CÁC PHẦN CỦA BÁO CÁO
# ==========================================

def show_profile(occupation, activities):
    section("[1] HỒ SƠ NGHỀ")
    code = occupation["occupation_code"]
    print(f"Nghề    : {occupation['occupation_title']}")
    print(f"Mã nghề : {code}")

    print("\nWork Activities nổi bật:")
    job = (
        activities[activities["occupation_code"] == code]
        .sort_values(["importance_score", "level_score"], ascending=False)
        .head(TOP_N_ACTIVITIES)
    )
    if job.empty:
        print("  (Không có dữ liệu Work Activities)")
        return
    for _, r in job.iterrows():
        print(f"  - {short(r['activity'], 50):<50} "
              f"Importance {r['importance_score']:.2f} | Level {r['level_score']:.2f}")


def show_clustering(code, activities, clusters, occupations):
    section("[2] CLUSTERING")
    assigned = clusters[clusters["occupation_code"] == code]
    if assigned.empty:
        print("Nghề này chưa có nhãn cluster trong CSV.")
        return None

    cluster_id = assigned.iloc[0]["cluster"]
    members = clusters.loc[clusters["cluster"] == cluster_id, "occupation_code"]
    print(f"Cụm được gán: Cluster {cluster_id}  ({len(members)}/{len(clusters)} nghề)")

    cluster_act = activities[activities["occupation_code"].isin(members)]
    cluster_mean = cluster_act.groupby("activity")["importance_score"].mean()
    overall_mean = activities.groupby("activity")["importance_score"].mean()

    print("\nWork Activities có Importance trung bình cao nhất của cụm:")
    for act, score in cluster_mean.sort_values(ascending=False).head(TOP_N_ACTIVITIES).items():
        print(f"  - {short(act, 56):<56} {score:.2f}")

    diff = (cluster_mean - overall_mean).sort_values(ascending=False)
    print("\nĐặc trưng riêng của cụm (cao hơn trung bình toàn bộ nghề):")
    for act, d in diff.head(5).items():
        print(f"  - {short(act, 56):<56} {d:+.2f}")
    print("Thấp hơn trung bình toàn bộ nghề:")
    for act, d in diff.tail(3).iloc[::-1].items():
        print(f"  - {short(act, 56):<56} {d:+.2f}")

    sample = occupations[occupations["occupation_code"].isin(members)]["occupation_title"].head(5)
    if not sample.empty:
        print("\nVí dụ nghề cùng cụm: " + "; ".join(short(t, 30) for t in sample))
    return cluster_id


def show_exposure(occupation, occupations, clusters, cluster_id):
    section("[3] AI EXPOSURE")
    exposure = occupation["ai_exposure_score"]
    if pd.isna(exposure):
        print("Nghề này không có AI Exposure score.")
        return

    valid = occupations.dropna(subset=["ai_exposure_score"])
    rank = (valid["ai_exposure_score"] > exposure).sum() + 1
    print(f"Điểm Exposure của nghề: {exposure:.4f}")
    print(f"Thứ hạng toàn bộ nghề : {rank}/{len(valid)} (cao đến thấp)")

    merged = clusters.merge(
        occupations[["occupation_code", "ai_exposure_score"]],
        on="occupation_code", how="left",
    )
    stats = (
        merged.dropna(subset=["ai_exposure_score"])
        .groupby("cluster")["ai_exposure_score"]
        .agg(["count", "mean", "median"])
        .sort_values("mean", ascending=False)
    )

    print("\nSo sánh với các cụm (Exposure trung bình, cao đến thấp):")
    for cid, row in stats.iterrows():
        mark = "  <-- cụm của nghề" if cid == cluster_id else ""
        print(f"  - Cluster {cid}: mean={row['mean']:.4f}, median={row['median']:.4f}, "
              f"n={int(row['count'])}{mark}")

    if cluster_id in stats.index:
        diff = exposure - stats.loc[cluster_id, "mean"]
        print(f"\nChênh lệch so với trung bình cụm: {diff:+.4f}")


def recommend(source_skill_code, matrix, z_matrix, top_n):
    raw = cosine_similarity(matrix.loc[[source_skill_code]], matrix)[0]
    zscore = cosine_similarity(z_matrix.loc[[source_skill_code]], z_matrix)[0]
    gaps = matrix.sub(matrix.loc[source_skill_code], axis="columns").abs().mean(axis=1)

    result = pd.DataFrame({
        "skill_code": matrix.index,
        "raw_cosine": raw,
        "zscore_cosine": zscore,
        "mean_abs_gap": gaps.values,
    })
    result = result[result["skill_code"] != source_skill_code].copy()
    result["sim_pct"] = result["zscore_cosine"].rank(pct=True)
    result["gap_pct"] = result["mean_abs_gap"].rank(pct=True)
    result["transition_score"] = 0.5 * result["sim_pct"] + 0.5 * (1 - result["gap_pct"])
    return (
        result.sort_values(["transition_score", "zscore_cosine"], ascending=False)
        .head(top_n).reset_index(drop=True)
    )


def show_transition(recs, title_map, exposure_map, cluster_map):
    section(f"[4] CAREER TRANSITION - TOP {len(recs)} NGHỀ ĐÍCH")
    print(f"{'#':<3}{'Mã nghề':<11}{'Nghề đích':<38}{'Score':>7}{'zCos':>8}{'Gap':>7}{'Cụm':>5}")
    for i, r in recs.iterrows():
        code = r["skill_code"]
        print(f"{i + 1:<3}{code:<11}{short(title_map.get(code, 'Unknown'), 36):<38}"
              f"{r['transition_score']:>7.3f}{r['zscore_cosine']:>8.3f}"
              f"{r['mean_abs_gap']:>7.3f}{str(cluster_map.get(code, '-')):>5}")
    print("\nScore = 0.5 x percentile độ giống (z-score cosine) + 0.5 x (1 - percentile gap trung bình).")


def classify_gap(source_skill_code, target_skill_code, matrix, skill_names):
    gap = pd.DataFrame({
        "skill_name": [skill_names.get(c, c) for c in matrix.columns],
        "source": matrix.loc[source_skill_code].values,
        "target": matrix.loc[target_skill_code].values,
    })
    gap["gap"] = gap["target"] - gap["source"]
    positive = gap[gap["gap"] > ZERO_GAP_TOL].sort_values("gap", ascending=False)
    negative = gap[gap["gap"] < -ZERO_GAP_TOL].sort_values("gap")
    zero = gap[gap["gap"].abs() <= ZERO_GAP_TOL].assign(
        abs_gap=lambda d: d["gap"].abs()
    ).sort_values("abs_gap")
    return positive, negative, zero


def print_gap_table(df, n):
    if df.empty:
        print("  (không có)")
        return
    print(f"  {'Kỹ năng':<40}{'Nguồn':>8}{'Đích':>8}{'Gap':>8}")
    for _, r in df.head(n).iterrows():
        print(f"  {short(r['skill_name'], 38):<40}{r['source']:>8.2f}"
              f"{r['target']:>8.2f}{r['gap']:>+8.2f}")


def show_skill_gap(src, tgt, matrix, skill_names, title_map):
    section("[5] SKILL GAP")
    print(f"{title_map.get(src, src)} ({src})")
    print(f"  -> {title_map.get(tgt, tgt)} ({tgt})")
    print(f"Gap = LV nghề đích - LV nghề nguồn; |gap| <= {ZERO_GAP_TOL} coi là bằng 0.")

    pos, neg, zero = classify_gap(src, tgt, matrix, skill_names)
    total = len(pos) + len(neg) + len(zero)
    print(f"Tổng {total} kỹ năng: {len(pos)} gap dương | {len(neg)} gap âm | {len(zero)} gap ≈ 0")

    print("\nGap dương (nghề đích đòi hỏi cao hơn, cần bổ sung):")
    print_gap_table(pos, TOP_N_SKILLS)
    print("\nGap âm (nghề nguồn đang mạnh hơn, lợi thế chuyển đổi):")
    print_gap_table(neg, TOP_N_SKILLS)
    print("\nGap ≈ 0 (đã tương đương, chuyển giao trực tiếp):")
    print_gap_table(zero, TOP_N_SKILLS)
    return pos, neg, zero


def show_commentary(src_code, source_title, source_cluster, source_exposure,
                    recs, title_map, exposure_map, cluster_map, gap_info, target_code):
    section("[6] NHẬN XÉT")

    n = len(recs)
    same_cluster = [c for c in recs["skill_code"] if source_cluster is not None
                    and cluster_map.get(c) == source_cluster]
    mean_z = recs["zscore_cosine"].mean()
    mean_gap = recs["mean_abs_gap"].mean()
    min_raw = recs["raw_cosine"].min()

    print("Vì sao các nghề được gợi ý:")
    print(f"  - Top {n} nghề có z-score cosine trung bình {mean_z:.3f} và gap kỹ năng "
          f"trung bình {mean_gap:.3f} (thang LV), tức hồ sơ kỹ năng gần '{short(source_title, 30)}'.")
    if source_cluster is not None:
        print(f"  - {len(same_cluster)}/{n} nghề đích cùng Cluster {source_cluster} với nghề nguồn, "
              "nghĩa là cấu trúc Work Activities cũng tương đồng." if same_cluster else
              f"  - Không nghề đích nào cùng Cluster {source_cluster}: kỹ năng gần nhau "
              "nhưng Work Activities khác cụm.")

    target_exp = [exposure_map.get(c) for c in recs["skill_code"]]
    target_exp = [e for e in target_exp if e is not None and not pd.isna(e)]
    if target_exp and source_exposure is not None and not pd.isna(source_exposure):
        lower = sum(e < source_exposure for e in target_exp)
        avg = sum(target_exp) / len(target_exp)
        print(f"  - AI Exposure: nguồn {source_exposure:.3f}, trung bình đích {avg:.3f}; "
              f"{lower}/{len(target_exp)} nghề đích có Exposure thấp hơn nguồn.")

    print("\nKết quả có hợp lý không:")
    verdicts = []
    if mean_z >= 0.5:
        verdicts.append("độ giống kỹ năng khá cao")
    else:
        verdicts.append("độ giống kỹ năng ở mức vừa phải, nên xem gợi ý như điểm khởi đầu")
    if source_cluster is not None and len(same_cluster) >= (n + 1) // 2:
        verdicts.append("đa số cùng cụm nên nhất quán với phân tích clustering")
    elif source_cluster is not None:
        verdicts.append("ít nghề cùng cụm, cần kiểm tra thủ công mức liên quan")
    print("  - " + "; ".join(verdicts).capitalize() + ".")
    if min_raw > 0.9:
        print(f"  - Raw cosine rất cao (thấp nhất {min_raw:.3f}) vì mọi nghề đều có điểm LV dương; "
              "vì vậy xếp hạng dựa trên z-score cosine và gap, không dựa vào raw cosine.")
    if gap_info is not None:
        pos, neg, zero = gap_info
        top_pos = pos.head(2)["skill_name"].tolist()
        msg = (f"  - Với nghề đích '{short(title_map.get(target_code, target_code), 30)}': "
               f"{len(pos)} kỹ năng cần nâng, {len(neg)} đã vượt, {len(zero)} tương đương.")
        print(msg)
        if top_pos:
            print(f"    Cần ưu tiên bổ sung: {', '.join(short(s, 30) for s in top_pos)}.")

    print("\nHạn chế:")
    for line in (
        "transition_score là điểm xếp hạng heuristic (trọng số 0.5/0.5), không phải xác suất chuyển nghề thành công.",
        "LV là hồ sơ kỹ năng ở cấp nghề O*NET, không đo năng lực cá nhân; chưa dùng Importance của kỹ năng.",
        "Chưa xét bằng cấp, Job Zone, mức lương, thị trường lao động hay địa phương (đặc biệt tại Việt Nam).",
        "K-Means K=4 khá thô, và AI Exposure là mức độ phơi nhiễm, không phải dự báo mất việc.",
        "Dữ liệu O*NET là ảnh chụp tại một thời điểm, các kỹ năng mới nổi có thể chưa được phản ánh.",
    ):
        print(f"  - {line}")


# ==========================================
# 7. LUỒNG CHÍNH
# ==========================================

def lookup(keyword, ctx, args, interactive):
    occupations, activities, clusters = ctx["occupations"], ctx["activities"], ctx["clusters"]
    matrix, z_matrix, skill_names = ctx["matrix"], ctx["z_matrix"], ctx["skill_names"]

    occupation = choose_occupation(keyword, occupations)
    if occupation is None:
        return

    code = occupation["occupation_code"]
    title = occupation["occupation_title"]
    exposure = occupation["ai_exposure_score"]

    banner(f"HỒ SƠ: {title} ({code})")
    show_profile(occupation, activities)
    cluster_id = show_clustering(code, activities, clusters, occupations)
    show_exposure(occupation, occupations, clusters, cluster_id)

    if matrix is None:
        section("[4] CAREER TRANSITION / [5] SKILL GAP")
        print(f"Bỏ qua do chưa tải được dữ liệu kỹ năng: {ctx['skill_error']}")
        return

    src_skill = resolve_code(code, matrix.index)
    if src_skill is None:
        section("[4] CAREER TRANSITION / [5] SKILL GAP")
        print(f"Mã nghề {code} không có trong dữ liệu kỹ năng LV O*NET.")
        return

    # Ánh xạ mã O*NET -> thông tin từ DB để hiển thị tiêu đề, exposure, cụm.
    title_map, exposure_map, cluster_map = {}, {}, {}
    for _, r in occupations.iterrows():
        sc = resolve_code(r["occupation_code"], matrix.index)
        if sc:
            title_map[sc] = r["occupation_title"]
            exposure_map[sc] = r["ai_exposure_score"]
    for _, r in clusters.iterrows():
        sc = resolve_code(r["occupation_code"], matrix.index)
        if sc:
            cluster_map[sc] = r["cluster"]

    recs = recommend(src_skill, matrix, z_matrix, args.top)
    show_transition(recs, title_map, exposure_map, cluster_map)

    target = args.target
    if target is None and interactive:
        pick = input(f"\nChọn nghề đích 1-{len(recs)} hoặc nhập mã nghề để xem skill gap "
                     "(Enter = 1): ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(recs):
            target = recs.loc[int(pick) - 1, "skill_code"]
        elif pick:
            target = pick
    if target is None:
        target = recs.loc[0, "skill_code"]

    tgt_skill = resolve_code(target, matrix.index)
    gap_info = None
    if tgt_skill is None:
        print(f"\nKhông tìm thấy nghề đích '{target}' trong dữ liệu kỹ năng.")
    else:
        gap_info = show_skill_gap(src_skill, tgt_skill, matrix, skill_names, title_map)

    show_commentary(src_skill, title, cluster_id, exposure, recs, title_map,
                    exposure_map, cluster_map, gap_info, tgt_skill)
    print("\n" + "=" * WIDTH)


def main():
    parser = argparse.ArgumentParser(description="AI Job Transition Map - terminal.")
    parser.add_argument("--code", help="Mã nghề hoặc từ khóa tên nghề, ví dụ 15-1252.00")
    parser.add_argument("--target", help="Mã nghề đích để xem skill gap (mặc định: top 1)")
    parser.add_argument("--top", type=int, default=TOP_N_TRANSITION,
                        help=f"Số nghề đích gợi ý (mặc định {TOP_N_TRANSITION})")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top phải >= 1")

    try:
        occupations, activities, clusters = load_db_data()
    except (FileNotFoundError, ValueError) as exc:
        print(f"LỖI DỮ LIỆU: {exc}", file=sys.stderr)
        return 1

    ctx = {"occupations": occupations, "activities": activities, "clusters": clusters,
           "matrix": None, "z_matrix": None, "skill_names": None, "skill_error": None}
    try:
        ctx["matrix"], ctx["z_matrix"], ctx["skill_names"] = load_skill_data()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        ctx["skill_error"] = str(exc)
        print(f"[Cảnh báo] Không tải được dữ liệu kỹ năng: {exc}", file=sys.stderr)

    banner("AI JOB TRANSITION MAP - TERMINAL")
    print(f"Số nghề trong database : {len(occupations)}")
    print(f"Số nghề có nhãn cluster: {len(clusters)}")
    if ctx["matrix"] is not None:
        print(f"Dữ liệu kỹ năng        : {len(ctx['matrix'])} nghề | "
              f"{ctx['matrix'].shape[1]} kỹ năng (LV)")

    if args.code:
        lookup(args.code, ctx, args, interactive=False)
        return 0

    while True:
        keyword = input("\nNhập mã nghề/tên nghề (q để thoát): ").strip()
        if keyword.lower() == "q":
            break
        if keyword:
            lookup(keyword, ctx, args, interactive=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())