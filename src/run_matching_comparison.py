import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sudachipy import dictionary, tokenizer
from tqdm import tqdm

# 出力ディレクトリの作成
os.makedirs("outputs", exist_ok=True)

# 1. データとモデルの読み込み
print("Word2Vecモデルを読み込んでいます...")
word_vectors = KeyedVectors.load("models/word2vec_vectors.kv")
print(f"語彙数: {len(word_vectors):,}")

print("データを読み込んでいます...")
project_df = pd.read_csv("data/raw/project-true.csv")
user_df = pd.read_csv("data/raw/user_work_histories.csv")
print(f"プロジェクト数: {len(project_df)}, ユーザー数: {len(user_df)}")


# 2. ヘルパー関数の定義
def extract_nouns(text: str) -> list[str]:
    if pd.isna(text):
        return []
    tokenizer_obj = dictionary.Dictionary(dict="full").create()
    mode = tokenizer.Tokenizer.SplitMode.C
    morphemes = tokenizer_obj.tokenize(str(text), mode)
    nouns = []
    for m in morphemes:
        pos = m.part_of_speech()[0]
        if pos == "名詞":
            nouns.append(m.surface())
    return nouns


def create_document_vector(text: str, word_vectors: KeyedVectors) -> np.ndarray:
    nouns = extract_nouns(text)
    vectors = []
    for noun in nouns:
        if noun in word_vectors:
            vectors.append(word_vectors[noun])
    if len(vectors) == 0:
        return np.zeros(word_vectors.vector_size)
    return np.mean(vectors, axis=0)


def create_project_vectors(
    project_df: pd.DataFrame, word_vectors: KeyedVectors
) -> np.ndarray:
    project_vectors = []
    for idx, row in tqdm(
        project_df.iterrows(), total=len(project_df), desc="プロジェクトベクトル作成"
    ):
        combined_text = f"{row['name']} {row['description']}"
        doc_vector = create_document_vector(combined_text, word_vectors)
        project_vectors.append(doc_vector)
    return np.array(project_vectors)


def match_candidates_for_project(
    project_idx, project_vectors, user_vectors, user_ids, top_k=10
):
    project_vector = project_vectors[project_idx].reshape(1, -1)
    similarities = cosine_similarity(project_vector, user_vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    candidates = [(user_ids[idx], similarities[idx]) for idx in sorted_indices]
    return candidates


def match_all_projects(project_df, project_vectors, user_vectors, user_ids, top_k=10):
    results = []
    for project_idx in tqdm(range(len(project_df)), desc="マッチング実行中"):
        candidates = match_candidates_for_project(
            project_idx, project_vectors, user_vectors, user_ids, top_k=top_k
        )
        for rank, (user_id, similarity) in enumerate(candidates, 1):
            results.append(
                {
                    "project_id": project_df.iloc[project_idx]["id"],
                    "project_name": project_df.iloc[project_idx]["name"],
                    "rank": rank,
                    "user_id": user_id,
                    "similarity_score": similarity,
                }
            )
    return pd.DataFrame(results)


# 3. プロジェクトベクトルの作成 (共通)
project_vectors = create_project_vectors(project_df, word_vectors)

# 4. 経歴のみのマッチング実行
print("\n--- 経歴のみのマッチング ---")


def create_user_profile_vectors_career(user_df, word_vectors):
    grouped = user_df.groupby("user_id")
    user_vectors = []
    user_ids = []
    for user_id, group in tqdm(grouped, desc="ユーザーベクトル作成 (経歴)"):
        all_texts = []
        for _, row in group.iterrows():
            combined_text = f"{row['project_name']} {row['description']} {row['role']}"
            all_texts.append(combined_text)
        user_text = " ".join(all_texts)
        user_vector = create_document_vector(user_text, word_vectors)
        user_vectors.append(user_vector)
        user_ids.append(user_id)
    return np.array(user_vectors), user_ids


user_vectors_career, user_ids_career = create_user_profile_vectors_career(
    user_df, word_vectors
)
results_career = match_all_projects(
    project_df, project_vectors, user_vectors_career, user_ids_career
)
results_career.to_csv(
    "outputs/matching_results_project_true.csv", index=False, encoding="utf-8-sig"
)
print("経歴のみのマッチング結果を保存しました。")

# 5. スキルのみのマッチング実行
print("\n--- スキルのみのマッチング ---")


def create_user_profile_vectors_skill(user_df, word_vectors):
    grouped = user_df.groupby("user_id")
    user_vectors = []
    user_ids = []
    skill_columns = [
        "HTML",
        "CSS",
        "JavaScript",
        "TypeScript",
        "Python",
        "Ruby",
        "PHP",
        "Java",
        "C#",
        "C++",
        "Go",
        "Rust",
        "Swift",
        "Kotlin",
        "React",
        "Vue",
        "Angular",
        "Node.js",
        "Express",
        "Django",
        "Flask",
        "Rails",
        "Laravel",
        "Spring",
        ".NET",
        "AWS",
        "Azure",
        "GCP",
        "Docker",
        "Kubernetes",
        "MySQL",
        "PostgreSQL",
        "MongoDB",
        "Git",
        "Linux",
        "Agile",
        "Scrum",
    ]
    available_skill_columns = [col for col in skill_columns if col in user_df.columns]

    for user_id, group in tqdm(grouped, desc="ユーザーベクトル作成 (スキル)"):
        user_skills = set()
        for _, row in group.iterrows():
            for col in available_skill_columns:
                if row[col] == 1:
                    user_skills.add(col)
        user_text = " ".join(user_skills)
        user_vector = create_document_vector(user_text, word_vectors)
        user_vectors.append(user_vector)
        user_ids.append(user_id)
    return np.array(user_vectors), user_ids


user_vectors_skill, user_ids_skill = create_user_profile_vectors_skill(
    user_df, word_vectors
)
results_skill = match_all_projects(
    project_df, project_vectors, user_vectors_skill, user_ids_skill
)
results_skill.to_csv(
    "outputs/matching_results_skill.csv", index=False, encoding="utf-8-sig"
)
print("スキルのみのマッチング結果を保存しました。")

# 6. 経歴とスキルの統合マッチング実行
print("\n--- 経歴とスキルの統合マッチング ---")


def create_user_profile_vectors_both(user_df, word_vectors):
    grouped = user_df.groupby("user_id")
    user_vectors = []
    user_ids = []
    skill_columns = [
        "HTML",
        "CSS",
        "JavaScript",
        "TypeScript",
        "Python",
        "Ruby",
        "PHP",
        "Java",
        "C#",
        "C++",
        "Go",
        "Rust",
        "Swift",
        "Kotlin",
        "React",
        "Vue",
        "Angular",
        "Node.js",
        "Express",
        "Django",
        "Flask",
        "Rails",
        "Laravel",
        "Spring",
        ".NET",
        "AWS",
        "Azure",
        "GCP",
        "Docker",
        "Kubernetes",
        "MySQL",
        "PostgreSQL",
        "MongoDB",
        "Git",
        "Linux",
        "Agile",
        "Scrum",
    ]
    available_skill_columns = [col for col in skill_columns if col in user_df.columns]

    for user_id, group in tqdm(grouped, desc="ユーザーベクトル作成 (統合)"):
        career_texts = []
        for _, row in group.iterrows():
            combined_text = f"{row['project_name']} {row['description']} {row['role']}"
            career_texts.append(combined_text)
        career_text = " ".join(career_texts)

        user_skills = set()
        for _, row in group.iterrows():
            for col in available_skill_columns:
                if row[col] == 1:
                    user_skills.add(col)
        skill_text = " ".join(user_skills)

        final_text = f"{career_text} {skill_text}"
        user_vector = create_document_vector(final_text, word_vectors)
        user_vectors.append(user_vector)
        user_ids.append(user_id)
    return np.array(user_vectors), user_ids


user_vectors_both, user_ids_both = create_user_profile_vectors_both(
    user_df, word_vectors
)
results_both = match_all_projects(
    project_df, project_vectors, user_vectors_both, user_ids_both
)
results_both.to_csv(
    "outputs/matching_results_both.csv", index=False, encoding="utf-8-sig"
)
print("統合マッチング結果を保存しました。")

# 7. 比較と分析
print("\n--- 結果の比較 ---")


def calculate_jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if len(s1.union(s2)) == 0:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))


project_ids = results_career["project_id"].unique()
similarities = []

for pid in tqdm(project_ids, desc="比較中"):
    users_career = (
        results_career[results_career["project_id"] == pid]["user_id"].head(10).tolist()
    )
    users_skill = (
        results_skill[results_skill["project_id"] == pid]["user_id"].head(10).tolist()
    )
    users_both = (
        results_both[results_both["project_id"] == pid]["user_id"].head(10).tolist()
    )

    sim_c_s = calculate_jaccard_similarity(users_career, users_skill)
    sim_c_b = calculate_jaccard_similarity(users_career, users_both)
    sim_s_b = calculate_jaccard_similarity(users_skill, users_both)

    similarities.append(
        {
            "project_id": pid,
            "career_vs_skill": sim_c_s,
            "career_vs_both": sim_c_b,
            "skill_vs_both": sim_s_b,
        }
    )

df_sim = pd.DataFrame(similarities)
print("\n【トップ10候補者のJaccard係数平均】")
print(df_sim.mean(numeric_only=True))

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_sim[["career_vs_skill", "career_vs_both", "skill_vs_both"]])
plt.title("Distribution of Jaccard Similarity (Top 10 Candidates)")
plt.ylabel("Jaccard Similarity")
plt.savefig("outputs/matching_comparison_boxplot.png")
print("比較箱ひげ図を保存しました: outputs/matching_comparison_boxplot.png")
