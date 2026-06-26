import cv2
from yolo_detector import YoloDetector

def main():
    # 通常のWebカメラ（内蔵カメラやUSBカメラ）を初期化
    # 0 は標準のカメラIDです。カメラが複数ある場合は 1 や 2 に変更してください。
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。カメラが接続されているか確認してください。")
        return

    print("カメラを起動しました。推論を開始します。")
    print("終了するには映像ウィンドウを選択した状態で 'q' キーを押してください。")

    # YOLO検出器の初期化 (標準の軽量モデル yolo11n.pt を使用)
    detector = YoloDetector(model_path='yolo11n.pt', conf_threshold=0.5)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームの取得に失敗しました。")
                break

            # YOLOで物体検出
            annotated_image, detections = detector.detect(frame)

            # 結果の表示
            cv2.imshow('Webcam YOLO Detection', annotated_image)

            # キー入力の処理
            key = cv2.waitKey(1) & 0xFF
            
            # 's'キーで画像のみを保存
            if key == ord('s'):
                import os
                import datetime
                save_dir = 'yolo_assets/collected_images'
                os.makedirs(save_dir, exist_ok=True)
                
                # タイムスタンプでユニークなファイル名を作成
                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
                filepath = os.path.join(save_dir, filename)
                
                # 枠が描かれていない「生の画像(frame)」のみ保存
                cv2.imwrite(filepath, frame)
                print(f"📸 画像のみを保存しました: {filepath}")

            # 'a'キーで画像とアノテーションの両方を保存（オートアノテーション）
            elif key == ord('a'):
                import os
                import datetime
                save_dir = 'yolo_assets/collected_images'
                os.makedirs(save_dir, exist_ok=True)
                
                base_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                img_filepath = os.path.join(save_dir, base_filename + ".jpg")
                txt_filepath = os.path.join(save_dir, base_filename + ".txt")
                
                # [重要] アノテーション用には枠が描かれていない「生の画像(frame)」を保存します
                cv2.imwrite(img_filepath, frame)
                print(f"📸 画像を保存しました: {img_filepath}")

                # アノテーション結果をYOLOフォーマットで自動保存（オートアノテーション）
                img_h, img_w, _ = frame.shape
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
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
