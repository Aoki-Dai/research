import os

import pandas as pd
from sudachipy import dictionary, tokenizer

# Add parent directory to path if needed, or just rely on relative paths
# Assuming running from project root


def extract_nouns(text: str) -> list[str]:
    """
    テキストから名詞を抽出する関数
    """
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


def get_project_text(project_id: int, project_df: pd.DataFrame) -> str:
    """プロジェクトのテキストを取得"""
    project = project_df[project_df["id"] == project_id].iloc[0]
    return f"{project['name']} {project['description']}"


def get_user_text(user_id: int, user_df: pd.DataFrame, method: str) -> str:
    """
    手法に応じてユーザーのテキストを取得
    method: 'career', 'skill', 'both'
    """
    user_data = user_df[user_df["user_id"] == user_id]

    # スキルカラムの定義（簡易版）
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

    text_parts = []

    if method in ["career", "both"]:
        # 職歴テキスト
        for _, row in user_data.iterrows():
            text_parts.append(
                f"{row['project_name']} {row['description']} {row['role']}"
            )

    if method in ["skill", "both"]:
        # スキルテキスト
        user_skills = set()
        for _, row in user_data.iterrows():
            for col in available_skill_columns:
                if row[col] == 1:
                    user_skills.add(col)
        text_parts.append(" ".join(user_skills))

    return " ".join(text_parts)


def analyze_match_reason(
    project_id: int, user_id: int, method: str, project_df, user_df
):
    """
    マッチング理由の分析（共通キーワードの抽出）
    """
    print(f"--- Analysis for Project {project_id} vs User {user_id} ({method}) ---")

    # テキスト取得
    p_text = get_project_text(project_id, project_df)
    u_text = get_user_text(user_id, user_df, method)

    # 名詞抽出
    p_nouns = set(extract_nouns(p_text))
    u_nouns = set(extract_nouns(u_text))

    # 共通キーワード
    common_keywords = p_nouns.intersection(u_nouns)

    print(f"Project Keywords (Top 10): {list(p_nouns)[:10]}...")
    print(f"User Keywords (Top 10): {list(u_nouns)[:10]}...")
    print(f"\n【共通キーワード ({len(common_keywords)}語)】")
    print(", ".join(common_keywords))
    print("\n")
    return common_keywords


def main():
    print("Loading data...")
    try:
        project_df = pd.read_csv("data/raw/project-true.csv")
        user_df = pd.read_csv("data/raw/user_work_histories.csv")

        # Load matching results
        df_career = pd.read_csv("outputs/matching_results_project_true.csv")
        df_skill = pd.read_csv("outputs/matching_results_skill.csv")
        df_both = pd.read_csv("outputs/matching_results_both.csv")

    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Current working directory:", os.getcwd())
        return

    # Analyze Project 1
    target_project_id = 1
    print(f"Analyzing Project {target_project_id}...")

    # Get top matches
    top_career = df_career[df_career["project_id"] == target_project_id].iloc[0][
        "user_id"
    ]
    top_skill = df_skill[df_skill["project_id"] == target_project_id].iloc[0]["user_id"]
    top_both = df_both[df_both["project_id"] == target_project_id].iloc[0]["user_id"]

    print(f"Top Matches for Project {target_project_id}:")
    print(f"Career: User {top_career}")
    print(f"Skill: User {top_skill}")
    print(f"Both: User {top_both}\n")

    analyze_match_reason(target_project_id, top_career, "career", project_df, user_df)
    analyze_match_reason(target_project_id, top_skill, "skill", project_df, user_df)
    analyze_match_reason(target_project_id, top_both, "both", project_df, user_df)

    # Analyze Project 2 (Management System)
    target_project_id = 2
    print(f"Analyzing Project {target_project_id}...")

    top_career = df_career[df_career["project_id"] == target_project_id].iloc[0][
        "user_id"
    ]
    top_skill = df_skill[df_skill["project_id"] == target_project_id].iloc[0]["user_id"]
    top_both = df_both[df_both["project_id"] == target_project_id].iloc[0]["user_id"]

    print(f"Top Matches for Project {target_project_id}:")
    print(f"Career: User {top_career}")
    print(f"Skill: User {top_skill}")
    print(f"Both: User {top_both}\n")

    analyze_match_reason(target_project_id, top_career, "career", project_df, user_df)
    analyze_match_reason(target_project_id, top_skill, "skill", project_df, user_df)
    analyze_match_reason(target_project_id, top_both, "both", project_df, user_df)


if __name__ == "__main__":
    main()
