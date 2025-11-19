import math
import os
from collections import Counter

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sudachipy import dictionary, tokenizer
from tqdm import tqdm

# Word2Vecモデルの読み込み
print("Word2Vecモデルを読み込んでいます...")
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

model_path = os.path.join(project_root, "models", "word2vec_wiki40b_full.wordvectors")
word_vectors = KeyedVectors.load(model_path, mmap="r")
print(f"語彙数: {len(word_vectors):,}語")
print(f"ベクトル次元数: {word_vectors.vector_size}")

# データの読み込み
print("データを読み込んでいます...")
project_df = pd.read_csv(os.path.join(project_root, "data", "raw", "project.csv"))
user_df = pd.read_csv(
    os.path.join(project_root, "data", "raw", "user_work_histories.csv")
)


# テキスト処理関数
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


# IDFの計算
print("IDFを計算しています...")
all_docs = []
for _, row in project_df.iterrows():
    all_docs.append(f"{row['name']} {row['description']} {row['desired_role']}")

grouped_user = user_df.groupby("user_id")
for _, group in grouped_user:
    all_texts = []
    for _, row in group.iterrows():
        all_texts.append(f"{row['project_name']} {row['description']} {row['role']}")
    all_docs.append(" ".join(all_texts))

doc_freq = Counter()
total_docs = len(all_docs)

for doc in tqdm(all_docs, desc="文書頻度をカウント中"):
    nouns = set(extract_nouns(doc))  # 文書ごとのユニーク単語数をカウント
    for noun in nouns:
        doc_freq[noun] += 1

idf_dict = {}
for word, freq in doc_freq.items():
    idf_dict[word] = math.log(total_docs / (freq + 1)) + 1  # IDFのスムージング

print(f"{len(idf_dict)}語のIDFを計算しました。")


# IDF重み付けを用いたcreate_document_vectorの修正版
def create_document_vector(
    text: str, word_vectors: KeyedVectors, idf_dict: dict
) -> np.ndarray:
    nouns = extract_nouns(text)

    vectors = []
    weights = []
    for noun in nouns:
        if noun in word_vectors:
            vectors.append(word_vectors[noun])
            weights.append(
                idf_dict.get(noun, 1.0)
            )  # 見つからない場合はデフォルトの重み1.0

    if len(vectors) == 0:
        return np.zeros(word_vectors.vector_size)

    return np.average(vectors, axis=0, weights=weights)


# プロジェクトベクトルの作成
print("プロジェクトベクトルを作成しています...")
project_vectors = []
for idx, row in tqdm(
    project_df.iterrows(), total=len(project_df), desc="プロジェクトベクトル"
):
    combined_text = f"{row['name']} {row['description']} {row['desired_role']}"
    doc_vector = create_document_vector(combined_text, word_vectors, idf_dict)
    project_vectors.append(doc_vector)
project_vectors = np.array(project_vectors)

# ユーザープロファイルベクトルの作成
print("ユーザープロファイルベクトルを作成しています...")
grouped = user_df.groupby("user_id")
user_vectors = []
user_ids = []

for user_id, group in tqdm(grouped, desc="ユーザープロファイルベクトル"):
    all_texts = []
    for _, row in group.iterrows():
        combined_text = f"{row['project_name']} {row['description']} {row['role']}"
        all_texts.append(combined_text)

    user_text = " ".join(all_texts)
    user_vector = create_document_vector(user_text, word_vectors, idf_dict)

    user_vectors.append(user_vector)
    user_ids.append(user_id)
user_vectors = np.array(user_vectors)


# マッチング関数
def match_candidates_for_project(
    project_idx: int,
    project_vectors: np.ndarray,
    user_vectors: np.ndarray,
    user_ids: list[int],
    top_k: int = 10,
) -> list[tuple[int, float]]:
    project_vector = project_vectors[project_idx].reshape(1, -1)
    similarities = cosine_similarity(project_vector, user_vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    candidates = [(user_ids[idx], similarities[idx]) for idx in sorted_indices]
    return candidates


# マッチングの実行
print("マッチングを実行しています...")
results = []
for project_idx in tqdm(range(len(project_df)), desc="マッチング"):
    candidates = match_candidates_for_project(
        project_idx, project_vectors, user_vectors, user_ids, top_k=10
    )

    for rank, (user_id, similarity) in enumerate(candidates, 1):
        results.append(
            {
                "project_id": project_df.iloc[project_idx]["id"],
                "project_name": project_df.iloc[project_idx]["name"],
                "desired_role": project_df.iloc[project_idx]["desired_role"],
                "rank": rank,
                "user_id": user_id,
                "similarity_score": similarity,
            }
        )

matching_results = pd.DataFrame(results)
output_path = os.path.join(project_root, "outputs", "matching_results_idf.csv")
matching_results.to_csv(output_path, index=False)
print(f"マッチング結果を保存しました: {output_path}")

# ベースラインとの比較
baseline_path = os.path.join(project_root, "outputs", "matching_results_wiki.csv")
if os.path.exists(baseline_path):
    print("\nベースラインと比較しています...")
    baseline_df = pd.read_csv(baseline_path)

    # 各プロジェクトのトップ1マッチを比較
    baseline_top1 = baseline_df[baseline_df["rank"] == 1].set_index("project_id")
    new_top1 = matching_results[matching_results["rank"] == 1].set_index("project_id")

    changed_count = 0
    total_projects = len(baseline_top1)

    print(f"{'プロジェクトID':<10} {'ベースライン':<15} {'新規':<15} {'変更あり?'}")
    print("-" * 50)

    for project_id in baseline_top1.index:
        base_user = baseline_top1.loc[project_id, "user_id"]
        new_user = new_top1.loc[project_id, "user_id"]

        changed = base_user != new_user
        if changed:
            changed_count += 1

        if project_id <= 10:  # 簡潔にするため最初の10件を表示
            print(
                f"{project_id:<10} {base_user:<15} {new_user:<15} {'はい' if changed else 'いいえ'}"
            )

    print("-" * 50)
    print(
        f"トップ1ユーザーが変更されたプロジェクト数: {changed_count}/{total_projects} 件 ({changed_count / total_projects:.2%})"
    )
else:
    print("ベースラインファイルが見つかりません。比較をスキップします。")
