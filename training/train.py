import os
import sys
from ultralytics import YOLO

# プロジェクトルートディレクトリの設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def main():
    # 1. モデルの読み込み
    # 新規学習の場合は事前学習済みの軽量モデル（yolo11n.pt）をベースにします。
    # Minisforum TH50のスペックを考慮し、処理速度の速い nano(n) または small(s) モデルを推奨。
    model_path = os.path.join(project_root, 'yolo11n.pt')
    model = YOLO(model_path) 

    # 2. 学習の実行
    # data: データセット設定ファイルのパス
    # epochs: 学習を繰り返す回数 (最初は50~100程度で様子を見ます)
    # imgsz: 入力画像サイズ (デフォルトは640)
    # batch: バッチサイズ (CPU環境やメモリ不足の場合は減らしてください)
    # device: 'cpu' （もし強力なNVIDIA GPUを積んだ別のPCで学習できる場合は '0' などを指定します）
    # 
    # [!] 注意: TH50 (Intel CPU) でも学習は可能ですが、非常に時間がかかります。
    # 可能であれば、学習自体はGoogle Colab等のGPU環境で行い、推論(predict)をTH50で行う構成をお勧めします。
    
    print("学習を開始します...")
    results = model.train(
        data=os.path.join(project_root, 'dataset.yaml'),
        epochs=100,
        imgsz=640,
        batch=8,
        device='cpu', 
        project=os.path.join(project_root, 'yolo_assets/robocon_models'),
        name='custom_model_v1',
        exist_ok=True
    )

    print("学習が完了しました！")
    print("モデルは 'yolo_assets/robocon_models/custom_model_v1/weights/best.pt' に保存されています。")

if __name__ == '__main__':
    main()
