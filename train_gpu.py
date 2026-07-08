import os
from ultralytics import YOLO

def main():
    # 1. モデルの読み込み
    # 新規学習の場合は事前学習済みの軽量モデル（yolo11n.pt）をベースにします。
    model = YOLO('yolo11n.pt') 

    # 2. 学習の実行 (GPUを使用)
    # batch: GPUメモリ（12GB）に余裕があるため、16にして高速化を図ります。
    # device: 0 (RTX 3060を指定)
    
    print("GPUでの学習を開始します...")
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0, 
        project='yolo_assets/robocon_models',
        name='custom_model_v1',
        exist_ok=True
    )

    print("学習が完了しました！")
    print("モデルは 'yolo_assets/robocon_models/custom_model_v1/weights/best.pt' に保存されています。")

if __name__ == '__main__':
    main()
