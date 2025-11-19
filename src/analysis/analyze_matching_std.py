import csv
import math
import pathlib
import statistics


def load_csv(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 数値型に変換
            try:
                row["rank"] = int(row["rank"])
                row["similarity_score"] = float(row["similarity_score"])
                data.append(row)
            except ValueError:
                continue
    return data


# スクリプトのディレクトリを取得
base_dir = pathlib.Path(__file__).parent.parent.parent
output_dir = base_dir / "outputs"

try:
    data_idf = load_csv(output_dir / "matching_results_idf.csv")
    data_wiki = load_csv(output_dir / "matching_results_wiki.csv")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit(1)

# スコア取得のヘルパー
scores_idf = [d["similarity_score"] for d in data_idf]
scores_wiki = [d["similarity_score"] for d in data_wiki]


def print_stats(name, scores):
    print(f"\n### {name}")
    if not scores:
        print("No data")
        return
    print(f"Count: {len(scores)}")
    print(f"Mean: {statistics.mean(scores):.4f}")
    print(f"Median: {statistics.median(scores):.4f}")
    print(f"Min: {min(scores):.4f}")
    print(f"Max: {max(scores):.4f}")
    if len(scores) > 1:
        print(f"Std Dev: {statistics.stdev(scores):.4f}")


print("## 基本統計量 (Similarity Score)")
print_stats("Word2Vec + IDF", scores_idf)
print_stats("Word2Vec (Wiki)", scores_wiki)

# 相関
# キーでマップ: (project_id, desired_role, user_id)
map_idf = {
    (d["project_id"], d["desired_role"], d["user_id"]): d["similarity_score"]
    for d in data_idf
}
map_wiki = {
    (d["project_id"], d["desired_role"], d["user_id"]): d["similarity_score"]
    for d in data_wiki
}

common_keys = set(map_idf.keys()) & set(map_wiki.keys())
if common_keys:
    x = [map_idf[k] for k in common_keys]
    y = [map_wiki[k] for k in common_keys]

    # ピアソン相関
    n = len(x)
    if n > 1:
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = math.sqrt(
            sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)
        )
        correlation = numerator / denominator if denominator != 0 else 0
        print(f"\n## スコアの相関係数: {correlation:.4f}")

        # 差分の統計
        diffs = [xi - yi for xi, yi in zip(x, y)]
        print("\n## スコア分布の差分 (IDF - Wiki)")
        print(f"平均: {statistics.mean(diffs):.4f}")
        if len(diffs) > 1:
            print(f"標準偏差: {statistics.stdev(diffs):.4f}")


# Top K の重複
def get_top_k(data, k=10):
    # project_id, desired_role でグループ化
    groups = {}
    for row in data:
        if row["rank"] <= k:
            key = (row["project_id"], row["desired_role"])
            if key not in groups:
                groups[key] = set()
            groups[key].add(row["user_id"])
    return groups


top_k_idf = get_top_k(data_idf)
top_k_wiki = get_top_k(data_wiki)

overlaps = []
for key, set_idf in top_k_idf.items():
    if key in top_k_wiki:
        set_wiki = top_k_wiki[key]
        intersection = len(set_idf.intersection(set_wiki))
        union = len(set_idf.union(set_wiki))
        if union > 0:
            overlaps.append(intersection / union)

if overlaps:
    print(f"\n## Top 10 候補者の平均Jaccard係数: {statistics.mean(overlaps):.4f}")

# ランク変動の例
print("\n## ランク変動の例 (Top 5 in IDF)")
if data_idf:
    sample_project = data_idf[0]["project_id"]
    sample_role = data_idf[0]["desired_role"]

    print(f"Project ID: {sample_project}, Role: {sample_role}")
    print("| Rank (IDF) | User ID | Score (IDF) | Rank (Wiki) | Score (Wiki) |")
    print("|---|---|---|---|---|")

    # IDFでのこのプロジェクトの上位5件を取得
    top_5_idf = [
        d
        for d in data_idf
        if d["project_id"] == sample_project
        and d["desired_role"] == sample_role
        and d["rank"] <= 5
    ]
    top_5_idf.sort(key=lambda x: x["rank"])

    # Wiki用のルックアップを作成
    wiki_lookup = {
        d["user_id"]: d
        for d in data_wiki
        if d["project_id"] == sample_project and d["desired_role"] == sample_role
    }

    for row in top_5_idf:
        user_id = row["user_id"]
        wiki_row = wiki_lookup.get(user_id)
        wiki_rank = wiki_row["rank"] if wiki_row else "-"
        wiki_score = wiki_row["similarity_score"] if wiki_row else "-"
        print(
            f"| {row['rank']} | {user_id} | {row['similarity_score']:.4f} | {wiki_rank} | {wiki_score} |"
        )
