from yolo_detector import YoloDetector

def main():
    # 学習済みモデルのパス
    # ※Google Colab等からダウンロードしたモデルを使用する場合は、このパスを書き換えてください。
    model_path = 'runs/detect/robocon_models/custom_model_v1/weights/best.pt'
    
    print(f"モデル '{model_path}' を読み込んでいます...")
    detector = YoloDetector(model_path=model_path)

    # OpenVINO形式にエクスポートする
    detector.export_to_openvino()

if __name__ == '__main__':
    main()
