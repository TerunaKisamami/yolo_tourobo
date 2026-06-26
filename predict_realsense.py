import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import argparse
from yolo_detector import YoloDetector

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='RealSense YOLO Detection and Image Collection')
    parser.add_argument('--save_dir', type=str, default='yolo_assets/collected_images', 
                        help='画像を保存するディレクトリのパス (デフォルト: yolo_assets/collected_images)')
    args = parser.parse_args()

    # 保存先ディレクトリの作成と確認
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 画像保存先: {os.path.abspath(save_dir)}")

    # 1. RealSenseのパイプライン初期化
    pipeline = rs.pipeline()
    config = rs.config()

    # カラーとデプスストリームを有効化 
    # (TH50の負荷を考慮し、まずは解像度を640x480, 30fpsに設定)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        print("RealSenseを起動しています...")
        profile = pipeline.start(config)
    except Exception as e:
        print(f"RealSenseの起動に失敗しました。カメラが接続されているか確認してください。")
        print(f"エラー詳細: {e}")
        return

    # 2. YOLO検出器の初期化
    # マージされたカスタム学習済みモデル（best.pt）を使用します。
    detector = YoloDetector(model_path='yolo_assets/robocon_models/custom_model_v1/weights/best.pt', conf_threshold=0.5)

    print("推論を開始します。終了するには画面を選択した状態で 'q' キーを押してください。")
    print("画像を保存するには 's' キーを押してください。")

    try:
        while True:
            # 3. フレームの取得
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # RealSenseフレームをOpenCVで扱えるnumpy配列に変換
            color_image = np.asanyarray(color_frame.get_data())

            # 4. YOLOで物体検出
            annotated_image, detections = detector.detect(color_image)

            # 5. 結果の表示
            cv2.imshow('RealSense YOLO Detection', annotated_image)

            # キー入力の処理
            key = cv2.waitKey(1) & 0xFF
            
            # 's'キーで画像のみを保存
            if key == ord('s'):
                # タイムスタンプでユニークなファイル名を作成
                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
                filepath = os.path.join(save_dir, filename)
                
                # 枠が描かれていない「生の画像(color_image)」のみ保存
                cv2.imwrite(filepath, color_image)
                print(f"📸 画像のみを保存しました: {filepath}")

            # 'a'キーで画像とアノテーションの両方を保存（オートアノテーション）
            elif key == ord('a'):
                base_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                img_filepath = os.path.join(save_dir, base_filename + ".jpg")
                txt_filepath = os.path.join(save_dir, base_filename + ".txt")
                
                # [重要] アノテーション用には枠が描かれていない「生の画像(color_image)」を保存します
                cv2.imwrite(img_filepath, color_image)
                print(f"📸 画像を保存しました: {img_filepath}")

                # アノテーション結果をYOLOフォーマットで自動保存（オートアノテーション）
                img_h, img_w, _ = color_image.shape
                with open(txt_filepath, 'w') as f:
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        cls_id = det['class_id']
                        
                        # YOLOフォーマットに変換 (中心x, 中心y, 幅, 高さ を 0~1 に正規化)
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        bbox_w = (x2 - x1) / img_w
                        bbox_h = (y2 - y1) / img_h
                        
                        # クラスID 中心X 中心Y 幅 高さ の形式で書き込み
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")
                
                print(f"📝 アノテーションを自動保存しました: {txt_filepath}")

            # 'q'キーで終了
            elif key == ord('q'):
                break

    finally:
        # 終了処理
        print("終了処理を行っています...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
