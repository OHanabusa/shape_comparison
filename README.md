# Shape Comparison Tool

画像の形状（エッジ）を比較し，類似度を計算するためのPythonツールです．実験画像とシミュレーション結果の比較に特化しています．
特に，比較画像のサイズが異なる場合や実験画像の上下を切り取った画像で比較したい場合に有効になります．

## 比較例

### 入力画像と比較結果
| 実験画像 | シミュレーション画像 |　比較結果 |
|:--------:|:------------------:|:----------------:|
| <img src= "https://github.com/user-attachments/assets/3c40ce97-6591-46e6-b987-96c13f37b8ad" width = "100">|<img src="https://github.com/user-attachments/assets/c3b8630f-afd1-4e74-8dab-60f0273d5e1d" width ="200"> |![比較結果](https://github.com/user-attachments/assets/a6e0a075-0319-42d4-891b-3b07b5f9e0a3)|


上記の例では，実験画像とシミュレーション画像のエッジを抽出し，形状の類似度を計算しています．比較結果の画像では，両者のエッジを重ね合わせて視覚的に差異を確認することができます．

## 特徴

- 高度な画像前処理機能
- 複数の類似度指標を組み合わせた総合評価
- デバッグ用の可視化機能
- 画像の部分的な比較機能（クロップ機能）

## 必要条件

- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-image

## インストール

```bash
pip install opencv-python numpy scikit-image
```

## 使用方法

### 基本的な使用例
426行目以降

```python
# 画像パスの設定
image1_path = '実験画像.png'
image2_path = 'シミュレーション画像.png'

# 画像の前処理
edge_image1 = load_and_preprocess_image(image1_path)
edge_image2 = load_and_preprocess_image(image2_path)

# 前処理済み画像の読み込み
img1 = cv2.imread(edge_image1)
img2 = cv2.imread(edge_image2)

# 類似度の計算
similarity = calculate_similarity(img1, img2)
print(f"類似度: {similarity:.2f}%")
```

### 画像の一部分のみを比較する場合

```python
# 画像の上20%から下80%までを使用
crop_ratio = (0.2, 0.8)

edge_image1 = load_and_preprocess_image(image1_path, crop_ratio)
edge_image2 = load_and_preprocess_image(image2_path, crop_ratio)
```

## 主要な機能

### 画像前処理 (`load_and_preprocess_image`)
- グレースケール変換
- ノイズ除去（ガウシアンブラー）
- コントラスト強調（CLAHE）
- Cannyエッジ検出
- 最大輪郭の抽出

### 類似度計算 (`calculate_similarity`)
以下の4つの指標を組み合わせて総合的な類似度を算出します：

| 指標 | 重み | 説明 |
|------|------|------|
| 面積類似度 | 10% | 二値化画像の面積比 |
| 周長類似度 | 10% | エッジの長さの比較 |
| SSIM類似度 | 50% | 構造的類似性指標 |
| オーバーラップ類似度 | 30% | エッジの重なり具合 |

## 出力ファイル

処理の過程で以下のファイルが自動的に生成されます：

- `debug_[元画像名].png`: エッジ検出過程の可視化結果
- `edge_only_[元画像名].png`: エッジのみを抽出した画像
- クロップを指定した場合は，ファイル名に`_cropped`が付加されます

## 注意事項

- 入力画像のサイズは自動的に調整されます
- 画像の向きや位置が大きく異なる場合，比較精度が低下する可能性があります
- 類似度は0-100の範囲で出力され，100が完全一致を示します
- デバッグ用の画像は自動的に保存されます

## ライセンス

MITライセンス
