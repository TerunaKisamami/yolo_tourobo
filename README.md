# YOLO Test プロジェクト ドキュメント

このディレクトリ (`yolo_tourobo`) には、ロボコン向けにYOLOを用いた物体検出、データセット収集、学習を行うためのスクリプト群を用途ごとに整理して格納しています。

## ディレクトリ構成

本プロジェクトは以下の用途ごとにフォルダ分けされています。

* **`inference/`**: カメラ映像からのリアルタイム物体検出、デプス（距離）測定、画像・アノテーションの自動収集
* **`training/`**: 収集したデータセットを用いたYOLOモデルの学習
* **`utils/`**: モデル形式の変換（OpenVINO）、データの仕分け、デプス測定評価などの補助ツール

---

## 動作環境・依存関係
以下のライブラリが必要です（`requirements.txt` に記載）。
- `ultralytics` (YOLO本体)
- `opencv-python` (画像処理・カメラ表示)
- `pyrealsense2` (RealSenseカメラ制御用)
- `openvino` (Intel CPU上での推論高速化用)

インストール方法:
```bash
pip install -r requirements.txt
```

---

## 各プログラムの仕様と使い方

### 1. 推論・データ収集 (`inference/`)

#### [predict_realsense.py](file:///home/hatsu/Robobobo/yolo_tourobo/inference/predict_realsense.py)
Intel RealSense カメラを使用したリアルタイム物体検出と、データセット収集（オートアノテーション機能付き）を行うメインスクリプトです。

**使い方**:
```bash
python inference/predict_realsense.py [--save_dir 保存先] [--target 初期対象]
```
* **キー操作**:
  * `s`キー: 現在の生のフレーム画像のみ（`.jpg`）を `yolo_assets/collected_images/` に保存します（最初のアノテーション用）。
  * `a`キー: 現在の生のフレーム画像と、YOLOの推論結果を用いたアノテーションデータ（`.txt`）をセットで自動保存します（オートアノテーション用）。
  * `0`〜`4`キー: 撮影・アノテーション対象クラスを動的に切り替えます（`volleyball_pink`, `volleyball_cyan`, `plate`, `block_red`, `block_blue`）。
  * `q`キー: プログラムを終了します。

#### [predict_realsense_fast.py](file:///home/hatsu/Robobobo/yolo_tourobo/inference/predict_realsense_fast.py)
OpenVINO形式（`yolo11n_openvino_model/`）を使用し、Intel CPU上での推論速度を大幅に高速化したリアルタイム測定用スクリプトです。

**使い方**:
```bash
python inference/predict_realsense_fast.py
```

#### [yolo_detector.py](file:///home/hatsu/Robobobo/yolo_tourobo/inference/yolo_detector.py)
YOLOの推論処理をまとめたモジュール（クラス）です。外部プログラム（ROS2ノードなど）から呼び出して利用しやすく設計されています。

---

### 2. モデル学習 (`training/`)

#### [train.py](file:///home/hatsu/Robobobo/yolo_tourobo/training/train.py) / [train_gpu.py](file:///home/hatsu/Robobobo/yolo_tourobo/training/train_gpu.py)
独自データセットを用いてYOLOモデルを学習（ファインチューニング）させるためのスクリプトです。ローカルCPU向けとGPU向けの2つがあります。

**使い方**:
```bash
python training/train.py      # CPUでの学習 (TH50等)
python training/train_gpu.py  # GPUでの学習 (RTX 3060等)
```
* `dataset.yaml` の設定に基づき、軽量モデル `yolo11n.pt` をベースに学習を開始します。
* 学習結果のモデルは `yolo_assets/robocon_models/custom_model_v1/weights/best.pt` に保存されます。

#### [auto_model_learn.py](file:///home/hatsu/Robobobo/yolo_tourobo/training/auto_model_learn.py)
自動収集したデータのマージ、YOLOモデルの再学習、OpenVINO形式への再変換、使用済みデータの退避までを一貫して自動で行う再学習パイプラインスクリプトです。

**使い方**:
```bash
python training/auto_model_learn.py
```

---

### 3. ユーティリティツール (`utils/`)

* **[convert_openvino.py](file:///home/hatsu/Robobobo/yolo_tourobo/utils/convert_openvino.py)**: 学習済みモデルをOpenVINO形式へコンバートします。
* **[evaluate_3d_accuracy.py](file:///home/hatsu/Robobobo/yolo_tourobo/utils/evaluate_3d_accuracy.py)**: YOLOで検出したターゲットの3D座標 `(X, Y, Z)` をサンプリングし、実測値（真値）との誤差統計（MAE、標準偏差など）を自動算出します。
* **[measure_raw_depth.py](file:///home/hatsu/Robobobo/yolo_tourobo/utils/measure_raw_depth.py)**: RealSenseのデプス欠損対策と精度測定を行い、`depth_evaluation.json` に評価結果を保存します。
* **[move_to_trained.py](file:///home/hatsu/Robobobo/yolo_tourobo/utils/move_to_trained.py)**: 学習に使用した画像・アノテーションデータを `trained` フォルダへ退避させます。
* **[sort_images_by_date.py](file:///home/hatsu/Robobobo/yolo_tourobo/utils/sort_images_by_date.py)**: 収集した画像ファイルを日付ごとに自動仕分けします。

---
## データセット収集のワークフロー（おすすめ）
1. `python inference/predict_realsense.py` を起動し、対象物を映しながら `s` キーを押してデータを集めます。
2. 自動生成された `.txt` ファイルを確認・微修正し、データを `yolo_assets/datasets/robocon_data/` に配置します。
3. `python training/train.py`（または GPU環境の `train_gpu.py`）を実行して学習し、新しい `best.pt` を作成します。
4. `python utils/convert_openvino.py` でOpenVINO形式にエクスポートします。
5. 作成されたモデルを `inference/yolo_detector.py` に指定して `predict_realsense_fast.py` を動かし、精度をテストします。

