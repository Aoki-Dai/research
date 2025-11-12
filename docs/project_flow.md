# プロジェクト処理フロー図

## 概要
このプロジェクトでは、CSVデータからWord2Vecモデルを学習し、プロジェクトとユーザーのマッチングを行う。

## 処理フロー

![](../results/figures/マッチング処理のデータフロー図.drawio.svg)

## 詳細な処理ステップ

### 1. CSVデータ読み込み
**入力ファイル:**
- `data/raw/project.csv` - プロジェクト情報（1,000件）
  - カラム: id, name, description, desired_role, スキルフラグ（40種類）
- `data/raw/user_work_histories.csv` - ユーザー職歴（911件）
  - カラム: user_id, project_name, description, role

**処理内容:**
- Pandasでデータを読み込み
- 欠損値の確認
- データ構造の把握

---

### 2. 前処理
**処理内容:**

#### 2.1 形態素解析
- **ツール:** Sudachipy（full辞書）
- **モード:** SplitMode.C（最も粗い粒度）
- **対象テキスト:**
  - プロジェクト: name, description, desired_role
  - ユーザー職歴: project_name, description, role

#### 2.2 名詞抽出
- 品詞が「名詞」のトークンのみを抽出
- 結果: 195種類のユニーク単語
- 総単語数: 34,498語

#### 2.3 コーパス作成
- 各ドキュメント（プロジェクトまたはユーザー職歴）を名詞のリストに変換
- プロジェクトコーパス: 1,000件
- ユーザーコーパス: 911件
- **合計: 1,911件のドキュメント**

---

### 3. モデル学習
**アルゴリズム:** Word2Vec (Skip-gram)

**ハイパーパラメータ:**
```python
vector_size=100        # ベクトルの次元数
window=5               # コンテキストウィンドウ
min_count=2            # 最小出現回数
sg=1                   # Skip-gram方式
epochs=100             # エポック数
negative=5             # ネガティブサンプリング数
alpha=0.025            # 初期学習率
min_alpha=0.0001       # 最小学習率
seed=42                # 再現性確保
```

**学習結果:**
- 語彙数: 195語
- 各単語が100次元のベクトルで表現される

---

### 4. モデル保存
**出力ファイル:**

1. **word2vec_model.bin**
   - 完全なモデル（再学習可能）
   - 場所: `results/models/`

2. **word2vec_vectors.kv**
   - 単語ベクトルのみ（軽量版）
   - 推論専用

3. **word_vectors.csv**
   - 全単語ベクトルをCSV形式で保存
   - 形状: (195, 100)
   - 分析・可視化用

---

### 5. 可視化
- **手法:** t-SNE（2次元圧縮）
- 頻出単語上位50件を2次元空間にプロット
- 類似した単語が近くに配置される

---

### 6. マッチ度計算

---

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| 言語 | Python |
| 形態素解析 | Sudachipy (full辞書) |
| 単語埋め込み | Gensim Word2Vec |
| データ処理 | Pandas, NumPy |
| 可視化 | Matplotlib, japanize-matplotlib |
| 単語ベクトル可視化のための次元削減 | scikit-learn (t-SNE) |

---

## ファイル構成

```
research/
├── data/
│   └── raw/
│       ├── project.csv                    # プロジェクトデータ
│       └── user_work_histories.csv        # ユーザー職歴データ
├── notebooks/
│   ├── project_preprocessing.ipynb        # 前処理ノートブック
│   └── word2vec.ipynb                     # Word2Vec学習ノートブック
├── results/
│   └── models/
│       ├── word2vec_model.bin             # 学習済みモデル
│       ├── word2vec_vectors.kv            # 単語ベクトル
│       └── word_vectors.csv               # ベクトルCSV
└── docs/
    └── project_flow.md                    # このファイル
```

---

## 今後の展開

1. **プロジェクト-ユーザーマッチング**
   - プロジェクトの説明文とユーザーの職歴を比較
   - コサイン類似度でマッチングスコアを算出

2. **推薦システム**
   - ユーザーに最適なプロジェクトをTop-N推薦
   - プロジェクトに最適なユーザーを推薦

3. **モデル改善**
   - Doc2Vec、BERT等の他の埋め込み手法の検討
   - ハイパーパラメータチューニング
   - より大規模なコーパスでの学習
