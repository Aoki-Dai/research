## コンポーネント

### データ

- **`data/raw/project.csv`**  
  - プロジェクト ID (`id`)
  - プロジェクト名 (`name`)
  - プロジェクト説明 (`description`)
  - 求める役割 (`desired_role`)
  - 技術スキルフラグ列（`HTML`, `CSS`, `JavaScript`, `AWS`, `Docker` など）

- **`data/raw/user_work_histories.csv`**  
  - ユーザー ID (`user_id`)
  - プロジェクト名 (`project_name`)
  - 詳細説明 (`description`)
  - 役割 (`role`)
  - 技術スキルフラグ列（`HTML`, `CSS`, `JavaScript`, `AWS`, `Docker` など）

### 前処理(形態素解析)

- **`notebooks/01_user_work_histories-sudachi.ipynb`**  
  - 職歴テキスト（`project_name`, `description`, `role`）から名詞抽出
  - 名詞頻度の集計・可視化

- **`notebooks/02_project_preprocessing.ipynb`**  
  - プロジェクトテキスト（`name`, `description`, `desired_role`）から名詞抽出
  - 名詞頻度の集計・可視化

### word2vecモデル学習

- **`notebooks/03_word2vec.ipynb`**  
  - プロジェクト・ユーザー職歴のテキストから名詞コーパスを作成
    - プロジェクト側: `name`, `description`, `desired_role`  
    - ユーザー側: `project_name`, `description`, `role`
  - `gensim.Word2Vec` による単語分散表現ベクトルの学習
  - 結果を`results/models/` に保存
    - `word2vec_model.bin`（Word2Vec モデル本体）
    - `word2vec_vectors.kv`（KeyedVectors）
    - `word_vectors.csv`（全単語ベクトルの CSV）
  - t-SNE による単語ベクトルの可視化

### マッチング処理

- **`notebooks/04_matching.ipynb`**  
  - `word2vec_vectors.kv` の読み込み
  - 名詞抽出と単語ベクトル平均を行い、以下のベクトルを生成
    - プロジェクトベクトル（プロジェクト1件ごと）
    - ユーザープロファイルベクトル（ユーザー1人ごとに職歴を統合）
  - `sklearn.metrics.pairwise.cosine_similarity` を用いてコサイン類似度を計算
  - 該当プロジェクトと全ユーザーの類似度を計算し、上位 `top_k` 名を推薦
  - 出力結果を保存
    - `results/matching_results.csv`  


## 使用技術

### 言語・実行環境

- Python 3.13
- Jupyter Notebook

### ライブラリ

- **データ分析・可視化**  
  - `pandas`  
  - `matplotlib`

- **形態素解析**  
  - `sudachipy`（形態素解析）  
  - `sudachidict-full`（辞書）

- **機械学習・分散表現**  
  - `gensim`（Word2Vec モデル学習）  
  - `scikit-learn`
    - `t-SNE`（可視化のための次元削減）
    - `cosine_similarity`（コサイン類似度計算）

- **その他**  
  - `tqdm`（進捗バー）  
  - `ipykernel`（Notebook 実行環境）
  - `ruff`（リンター・フォーマッター）
