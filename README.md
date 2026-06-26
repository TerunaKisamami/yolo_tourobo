# YOLO Test プロジェクト ドキュメント

このディレクトリ (`yolo_test`) には、ロボコン向けにYOLOを用いた物体検出とデータセット収集を行うためのスクリプト群が含まれています。

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

## 各プログラムの仕様と使い方

### 1. `predict_realsense.py`
Intel RealSense カメラを使用したリアルタイム物体検出と、データセット収集（オートアノテーション機能付き）を行うスクリプトです。

**使い方**:
```bash
python predict_realsense.py [--save_dir 保存先ディレクトリ]
```
- RealSenseカメラの映像を取得し、リアルタイムでYOLOによる推論結果を画面に表示します。
- **キー操作**:
  - `s`キー: 現在のフレーム画像のみ（`.jpg`）を `yolo_assets/collected_images` に保存します（最初の手作業アノテーション用）。
  - `a`キー: 現在のフレーム画像と、YOLOの推論結果を用いたアノテーションデータ（`.txt`）をセットで自動保存します（オートアノテーション用）。
  - `q`キー: プログラムを終了します。

### 2. `predict_webcam.py`
一般的なWebカメラ（内蔵カメラやUSBカメラ）を使用したリアルタイム物体検出とデータ収集を行うスクリプトです。RealSenseが接続されていない環境でのテスト等に有用です。

**使い方**:
```bash
python predict_webcam.py
```
- **キー操作**:
  - `s`キー: 現在のフレーム画像のみ（`.jpg`）を `yolo_assets/collected_images` に保存します。
  - `a`キー: 現在のフレーム画像と、YOLOの推論結果を用いたアノテーションデータ（`.txt`）をセットで自動保存します。
  - `q`キー: プログラムを終了します。

### 3. `yolo_detector.py`
YOLOの推論処理をまとめたモジュール（クラス）です。外部プログラム（ROS2ノードなど）から呼び出して利用しやすく設計されています。

**主な機能**:
- `YoloDetector(model_path, conf_threshold)`: 検出器の初期化。標準では `yolo11n.pt` が使用されます。
- `detect(frame)`: OpenCV画像を入力として受け取り、バウンディングボックスが描画された画像と、検出結果のメタデータ（クラスID、名前、信頼度、座標）のリストを返します。
- `export_to_openvino()`: Intel CPU (TH50など) での推論速度を向上させるため、モデルをOpenVINO形式にエクスポートします。

### 4. `train.py`
独自データセットを用いてYOLOモデルを学習（ファインチューニング）させるためのスクリプトです。

**使い方**:
```bash
python train.py
```
- `dataset.yaml` の設定に基づき、軽量モデル `yolo11n.pt` をベースに学習を開始します。
- デフォルトではCPU環境での学習になるよう設定されていますが、学習には非常に時間がかかるため、本格的な学習はGoogle Colab等のGPU環境で行うことが推奨されています。
- 学習結果のモデルは `yolo_assets/robocon_models/custom_model_v1/weights/best.pt` に保存されます。

### 5. `dataset.yaml`
YOLO学習用のデータセット構成ファイルです。
- `path`: データセットのルートディレクトリ
- `train`, `val`: 学習用・検証用画像のパス
- `nc`: クラス数
- `names`: クラス名のマッピング (例: `0: volleyball_cyan`, `1: volleyball_pink`, `2: plate`)

---
## データセット収集のワークフロー（おすすめ）
1. `predict_realsense.py` を起動し、対象物を様々な角度や環境下で映しながら `s` キーを押してデータを集めます。
2. 自動生成された `.txt` ファイルを確認し、必要に応じて「labelImg」や「Roboflow」などのアノテーションツールで誤検出箇所を微修正します（完全にゼロからアノテーションするより大幅に時間を短縮できます）。
3. 修正したデータと画像を `dataset.yaml` で指定したディレクトリ配置に整理し、`train.py`（または外部のGPU環境）で学習を実行して新しいモデル(`best.pt`)を作成します。
4. 作成された `best.pt` を `yolo_detector.py` の読み込み元に指定し、再度 `predict_realsense.py` や `predict_webcam.py` を起動して検出精度をテストします。
