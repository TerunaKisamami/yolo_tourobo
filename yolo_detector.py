import cv2
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.5):
        """
        YOLO検出器の初期化。ROS2ノード等の外部システムから呼び出しやすいようクラス化しています。
        
        :param model_path: 学習済みモデルのパス (例: best.pt)
                           Intel CPU (TH50) 上で高速化する場合は、事前にOpenVINO形式
                           (best_openvino_model/)にエクスポートしたパスを指定するとより高速になります。
        :param conf_threshold: 信頼度の閾値 (これ以下の確率の検出は無視する)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        画像フレームから物体検出を行う
        
        :param frame: OpenCV画像 (numpy array)
        :return: 描画済みの画像, 検出結果のリスト (ROS2メッセージ化しやすい形)
        """
        # 推論の実行
        # verbose=False で標準出力を減らし、ログを綺麗にします
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # 検出結果のバウンディングボックスを描画した画像を生成
        annotated_frame = results[0].plot()

        # 検出結果のメタデータを抽出
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist() # バウンディングボックスの座標
            conf = float(box.conf[0])             # 信頼度
            cls_id = int(box.cls[0])              # クラスID
            cls_name = self.model.names[cls_id]   # クラス名
            
            detections.append({
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })

        return annotated_frame, detections

    def export_to_openvino(self):
        """
        Intel CPU 向けにモデルをOpenVINO形式にエクスポートして推論速度を向上させます。
        初回のみ実行し、以降はエクスポートされたディレクトリをモデルパスに指定して初期化します。
        """
        print("OpenVINO形式へエクスポートしています...")
        self.model.export(format='openvino')
        print("エクスポートが完了しました。")

if __name__ == '__main__':
    print("==========================================================")
    print("yolo_detector.py は物体検出（推論）用のモジュールクラスです。")
    print("直接実行してもモデルの学習や `best.pt` の生成は行われません。")
    print("==========================================================\n")
    print("【モデルの学習（best.ptの作成）を行いたい場合】")
    print("  python train.py を実行してください。")
    print("  学習完了後、'yolo_assets/robocon_models/custom_model_v1/weights/best.pt' に保存されます。")
    print("\n【カメラによるリアルタイム検出を試したい場合】")
    print("  python predict_realsense.py を実行してください。")

