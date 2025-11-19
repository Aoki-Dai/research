import numpy as np
import pandas as pd

# データの読み込み
try:
    df_idf = pd.read_csv("outputs/matching_results_idf.csv")
    df_wiki = pd.read_csv("outputs/matching_results_wiki.csv")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit(1)

# 1. 基本統計量
print("## 基本統計量 (Similarity Score)")
print("\n### Word2Vec + IDF")
print(df_idf["similarity_score"].describe().to_markdown())
print("\n### Word2Vec (Wiki)")
print(df_wiki["similarity_score"].describe().to_markdown())

# 2. 相関
# project_id, desired_role, user_id でマージ
merged = pd.merge(
    df_idf,
    df_wiki,
    on=["project_id", "desired_role", "user_id"],
    suffixes=("_idf", "_wiki"),
)
correlation = merged["similarity_score_idf"].corr(merged["similarity_score_wiki"])
print(f"\n## スコアの相関係数: {correlation:.4f}")


# 3. Top K の重複 (Top 10 候補者のJaccard係数)
def get_top_k_users(df, k=10):
    return (
        df[df["rank"] <= k]
        .groupby(["project_id", "desired_role"])["user_id"]
        .apply(set)
    )


top_k_idf = get_top_k_users(df_idf)
top_k_wiki = get_top_k_users(df_wiki)

overlaps = []
for index in top_k_idf.index:
    if index in top_k_wiki.index:
        set_idf = top_k_idf[index]
        set_wiki = top_k_wiki[index]
        intersection = len(set_idf.intersection(set_wiki))
        union = len(set_idf.union(set_wiki))
        if union > 0:
            overlaps.append(intersection / union)

if overlaps:
    avg_overlap = np.mean(overlaps)
    print(f"\n## Top 10 候補者の平均Jaccard係数: {avg_overlap:.4f}")
else:
    print("\n## Top 10 候補者の平均Jaccard係数: N/A")

# 4. 上位ユーザーのランク変動
# サンプルプロジェクトにおけるIDFの上位5ユーザーのWikiでの順位を確認
print("\n## ランク変動の例 (Top 5 in IDF)")
if not df_idf.empty:
    sample_project = df_idf["project_id"].unique()[0]
    # このプロジェクトの最初の役割を取得
    roles = df_idf[df_idf["project_id"] == sample_project]["desired_role"].unique()
    if len(roles) > 0:
        sample_role = roles[0]

        print(f"Project ID: {sample_project}, Role: {sample_role}")
        top_idf_sample = df_idf[
            (df_idf["project_id"] == sample_project)
            & (df_idf["desired_role"] == sample_role)
            & (df_idf["rank"] <= 5)
        ]
        top_wiki_sample = df_wiki[
            (df_wiki["project_id"] == sample_project)
            & (df_wiki["desired_role"] == sample_role)
        ]

        merged_sample = pd.merge(
            top_idf_sample,
            top_wiki_sample[["user_id", "rank", "similarity_score"]],
            on="user_id",
            suffixes=("_idf", "_wiki"),
        )
        print(
            merged_sample[
                [
                    "rank_idf",
                    "user_id",
                    "similarity_score_idf",
                    "rank_wiki",
                    "similarity_score_wiki",
                ]
            ].to_markdown(index=False)
        )

# 5. 分布の差分
print("\n## スコア分布の差分")
diff = merged["similarity_score_idf"] - merged["similarity_score_wiki"]
print(f"IDFスコア - Wikiスコア 平均: {diff.mean():.4f}")
print(f"IDFスコア - Wikiスコア 標準偏差: {diff.std():.4f}")
