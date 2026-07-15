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

#### [predict_realsense_high_accuracy.py](file:///home/hatsu/Robobobo/yolo_tourobo/inference/predict_realsense_high_accuracy.py)
フィルタの最大化と移動平均バッファにより、測距のブレやノイズを極限まで抑えた高精度評価用の推論スクリプトです（少し残像が発生します）。

**使い方**:
```bash
python inference/predict_realsense_high_accuracy.py [--target ターゲットクラス名]
```

#### [yolo_detector.py](file:///home/hatsu/Robobobo/yolo_tourobo/inference/yolo_detector.py)
YOLOの推論処理をまとめたモジュール（クラス）です。外部プログラム（ROS2ノードなど）から呼び出して利用しやすく設計されています。

---

### 2. モデル学習 (`training/`)

#### [train.py](file:///home/hatsu/Robobobo/yolo_tourobo/training/train.py) / [train_gpu.py](file:///home/hatsu/Robobobo/yolo_tourobo/training/train_gpu.py)
独自データセットを用いてYOLOモデルを学習（ファインチューニング）させるためのスクリプトです。ローカルCPU向けとGPU向けの2つがあります。

**使い方**:
```bash
python training/train.py      # CPUでの学習
python training/train_gpu.py  # GPUでの学習 (CUDA環境)
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

#### [convert_openvino.py](file:///home/hatsu/Robobobo/yolo_tourobo/utils/convert_openvino.py)
学習済みのPyTorchモデル（`.pt`）を、Intel CPUでの高速動作に適したOpenVINO形式モデルへコンバートします。

**使い方**:
```bash
python utils/convert_openvino.py
```
* 変換前のモデルとして `yolo_assets/robocon_models/custom_model_v1/weights/best.pt` を読み込み、`yolo11n_openvino_model/` に出力します。

#### [evaluate_3d_accuracy.py](file:///home/hatsu/Robobobo/yolo_tourobo/utils/evaluate_3d_accuracy.py)
YOLOによる3D座標予測の誤差（真値とのズレ）を評価するための精度測定プログラムです。

**使い方**:
```bash
python utils/evaluate_3d_accuracy.py --target [クラス名] --x_true [X真値mm] --y_true [Y真値mm] --z_true [Z真値mm] [--frames サンプリング数] [--mock]
```
* **オプション**:
  * `--target`: 対象クラス名（例: `block_red`, `volleyball_pink` 等）
  * `--x_true`, `--y_true`, `--z_true`: テープメジャー等で実測したカメラからターゲット中心までの3次元の距離（ミリメートル単位）
  * `--frames`: 平均化に使用するサンプリングフレーム数（デフォルト: 100）
  * `--mock`: カメラが接続されていない状態でシミュレーション評価を実行するテストモード
* 評価結果は `utils/3d_accuracy_evaluation.json` に統計情報（平均誤差、標準偏差など）として保存されます。

#### [generate_synthetic_dataset.py](file:///home/hatsu/Robobobo/yolo_tourobo/utils/generate_synthetic_dataset.py)
透過PNG画像（前景）と背景画像をランダムに合成し、アノテーションデータ（`.txt`）を自動生成するCGコラージュデータセット作成ツールです。

**使い方**:
```bash
python utils/generate_synthetic_dataset.py [--fg_dir 前景フォルダ] [--bg_dir 背景フォルダ] [--output_dir 出力フォルダ] [--num_images 生成枚数] [--class_id クラスID] [--val_split 検証データの割合]
```
* **主なオプション**:
  * `--fg_dir`: 前景となる透過PNG画像のフォルダ（デフォルト: `yolo_assets/synthetic_sources/foregrounds`）
  * `--bg_dir`: 背景画像のフォルダ（デフォルト: `yolo_assets/synthetic_sources/backgrounds`）
  * `--output_dir`: 出力先データセットフォルダ（デフォルト: `yolo_assets/datasets/synthetic_dataset`）
  * `--num_images`: 生成する画像・ラベルのセット数（デフォルト: 1000）
  * `--class_id`: 合成する物体のYOLOクラスID。`dataset.yaml`に対応する番号を指定（デフォルト: 3 (`block_red`)）
  * `--val_split`: validationデータとして分ける割合（`0.0 ~ 1.0`）（デフォルト: 0.2）

#### [measure_raw_depth.py](file:///home/hatsu/Robobobo/yolo_tourobo/utils/measure_raw_depth.py)
RealSenseカメラの画面中央（赤十字）のデプス（距離）測定とノイズ評価を単体で行うツールです。

**使い方**:
```bash
python utils/measure_raw_depth.py
```
* **キー操作**:
  * `r`キー: 100フレーム分の距離データをサンプリングし、評価データとして `depth_evaluation.json` に保存します。
  * `q`キー: プログラムを終了します。

#### [move_to_trained.py](file:///home/hatsu/Robobobo/yolo_tourobo/utils/move_to_trained.py)
モデルの学習が完了した画像データを、`trained` サブフォルダへ一括退避させてデータセット収集ディレクトリをクリーンアップします。

**使い方**:
```bash
python utils/move_to_trained.py
```
* `yolo_assets/collected_images/` 配下の指定構造ファイルを移動します。

#### [sort_images_by_date.py](file:///home/hatsu/Robobobo/yolo_tourobo/utils/sort_images_by_date.py)
収集されたデータ群のうち、日付別に整理されていない画像やアノテーションを、ファイル名のタイムスタンプ日付に基づいて `YYYY-MM-DD` 形式のフォルダへ自動仕分けします。

**使い方**:
```bash
python utils/sort_images_by_date.py
```

---
## データセット収集・学習のワークフロー（実機のみの場合）
1. `python inference/predict_realsense.py` を起動し、対象物を映しながら `s` キーを押してデータを集めます。
2. 自動生成された `.txt` ファイルを確認・微修正し、データを `yolo_assets/datasets/robocon_data/` に配置します。
3. `python training/train.py`（または GPU環境の `train_gpu.py`）を実行して学習し、新しい `best.pt` を作成します。
4. `python utils/convert_openvino.py` でOpenVINO形式にエクスポートします。
5. 作成されたモデルを `inference/yolo_detector.py` に指定して `predict_realsense_fast.py` を動かし、精度をテストします。

---
## 合成データ（コラージュ）を用いたデータ収集・自動アノテーションフロー
1. **素材の用意**:
   * 背景を透過した箱の画像（`.png`）を `yolo_assets/synthetic_sources/foregrounds/` に配置します。
   * 箱の背景となる床やフィールドの画像（`.jpg`/`.png`）を `yolo_assets/synthetic_sources/backgrounds/` に配置します。
2. **合成データの生成**:
   以下のコマンドを実行し、合成データセットを作成します。
   ```bash
   python utils/generate_synthetic_dataset.py --num_images 1000 --class_id 3 --output_dir yolo_assets/datasets/synthetic_dataset
   ```
   ※ `--class_id` にはアノテーションしたい対象（3: block_red, 4: block_blue など）を指定します。その他のオプションは `python utils/generate_synthetic_dataset.py --help` で確認できます。
3. **初期YOLOモデルの学習**:
   生成された `synthetic_dataset` のパスを `dataset.yaml` に設定し、`training/train.py` で学習を行って初期YOLOモデルを作成します。
4. **実画像に対する自動アノテーション**:
   実機カメラで撮影した未アノテーションの実画像に対し、初期YOLOモデルで推論を行い、得られたバウンディングボックスを元にSAM（Segment Anything Model）などを用いて自動で高精度なアノテーション（`.txt`）を生成します。
5. **人間によるチェックと最終学習**:
   アノテーションツール（CVATやLabel Studio等）上で、自動生成されたアノテーションのズレや間違いがないかを人間がチェック・修正します。
   修正済みの高品質なデータセットで再学習し、最終モデルを作成します。

