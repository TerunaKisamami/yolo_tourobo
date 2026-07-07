import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import argparse
import time
from yolo_detector import YoloDetector

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='RealSense YOLO Detection and Image Collection')
    parser.add_argument('--save_dir', type=str, default='yolo_assets/collected_images', 
                        help='基本の保存ディレクトリパス (デフォルト: yolo_assets/collected_images)')
    parser.add_argument('--target', type=str, default='block_red',
                        help='初期の撮影対象名。実行中に0〜4キーで変更可能です。')
    args = parser.parse_args()

    current_target = args.target
    target_classes = {
        ord('0'): 'volleyball_pink',
        ord('1'): 'volleyball_cyan',
        ord('2'): 'plate',
        ord('3'): 'block_red',
        ord('4'): 'block_blue',
    }

    # 各オブジェクトのデプス取得モード（実験用設定）
    # 'center' : 中心点付近の中央値を取得 (ボールやブロックなど立体向け)
    # 'near'   : 枠内の最近点(上位5%)を取得 (皿などパースで中心がズレやすい薄い物体向け)
    depth_modes = {
        'volleyball_pink': 'near',
        'volleyball_cyan': 'center',
        'plate': 'near',
        'block_red': 'center',
        'block_blue': 'center',
    }

    # 1. RealSenseのパイプライン初期化
    pipeline = rs.pipeline()
    config = rs.config()

    # カラーとデプスストリームを有効化 
    # (TH50の負荷を考慮し、まずは解像度を640x480, 30fpsに設定)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # カラーとデプスの位置合わせ用オブジェクト
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 2. YOLO検出器の初期化
    # マージされたカスタム学習済みモデル（best.pt）を使用します。
    detector = YoloDetector(model_path='yolo_assets/robocon_models/custom_model_v1/weights/best.pt', conf_threshold=0.5)

    # RealSense SDK 組み込みフィルターの初期化（デプス欠損ゼロ対策）
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    hole_filling_filter = rs.hole_filling_filter()

    try:
        print("RealSenseを起動しています...")
        profile = pipeline.start(config)
    except Exception as e:
        print(f"RealSenseの起動に失敗しました。カメラが接続されているか確認してください。")
        print(f"エラー詳細: {e}")
        return

    print("推論を開始します。終了するには画面を選択した状態で 'q' キーを押してください。")
    print("画像を保存するには 's' キーを押してください。")

    try:
        prev_time = time.time()
        while True:
            # 3. フレームの取得と位置合わせ
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # デプスフレームにSDKフィルターを適用（欠損穴埋め・ノイズ除去）
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)
            depth_frame = hole_filling_filter.process(depth_frame).as_depth_frame()

            # RealSenseフレームをOpenCVで扱えるnumpy配列に変換
            color_image = np.asanyarray(color_frame.get_data())

            # 4. YOLOで物体検出
            annotated_image, detections = detector.detect(color_image)

            # 3D座標の計算と描画
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cls_name = det.get('class_name', '')
                
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                # インデックスの範囲外アクセスを防ぐためにクリップ
                cx = max(0, min(cx, color_frame.width - 1))
                cy = max(0, min(cy, color_frame.height - 1))

                # ユーザーが設定した辞書(depth_modes)に基づいて取得モードを決定
                current_mode = depth_modes.get(cls_name, 'center')
                mode_closest = (current_mode == 'near')

                valid_distances = []
                
                if mode_closest:
                    # 最近点取得モード (案3: 枠内の縮小領域から最小に近い距離を探す)
                    margin_x = int((x2 - x1) * 0.15)
                    margin_y = int((y2 - y1) * 0.15)
                    search_x1 = max(0, int(x1) + margin_x)
                    search_x2 = min(color_frame.width - 1, int(x2) - margin_x)
                    search_y1 = max(0, int(y1) + margin_y)
                    search_y2 = min(color_frame.height - 1, int(y2) - margin_y)

                    # 高速化のため3ピクセル飛ばしで検索
                    for py in range(search_y1, search_y2 + 1, 3):
                        for px in range(search_x1, search_x2 + 1, 3):
                            dist = depth_frame.get_distance(px, py)
                            if dist > 0:
                                valid_distances.append(dist)
                else:
                    # 中心点取得モード（今まで通り 5x5 ピクセル）
                    half_w = 2
                    for dy in range(-half_w, half_w + 1):
                        for dx in range(-half_w, half_w + 1):
                            px, py = cx + dx, cy + dy
                            if 0 <= px < color_frame.width and 0 <= py < color_frame.height:
                                dist = depth_frame.get_distance(px, py)
                                if dist > 0:
                                    valid_distances.append(dist)
                
                if len(valid_distances) > 0:
                    if mode_closest:
                        # 案3: 極端な外れ値（ノイズ）を避けるため、上位5%（パーセンタイル）を最近点として扱う
                        distance = np.percentile(valid_distances, 5)
                        color = (0, 165, 255) # 違いがわかるように色を変える (オレンジ)
                        mode_text = "Near"
                    else:
                        distance = np.median(valid_distances)
                        color = (0, 255, 0) # 今まで通り (緑)
                        mode_text = "Center"
                        
                    # 3D座標の計算
                    x_val, y_val, z_val = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], distance)
                    
                    # 画面表示用にミリメートルに変換して文字列を作成
                    text = f"Dist({mode_text}): {distance:.2f}m ({x_val*1000:.0f}, {y_val*1000:.0f}, {z_val*1000:.0f})mm"
                    
                    # 中心点（赤丸）とテキストを画面に描画
                    cv2.circle(annotated_image, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(annotated_image, text, (int(x1), int(y2) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 時間計測とFPS計算
            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            fps = 1.0 / exec_time if exec_time > 0 else 0.0

            # 5. 結果の表示
            cv2.putText(annotated_image, f"Target: {current_target} (Change: 0-4)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_image, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            cv2.imshow('RealSense YOLO Detection', annotated_image)

            # キー入力の処理
            key = cv2.waitKey(1) & 0xFF
            
            # 's'キーで画像のみを保存
            if key == ord('s'):
                date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                base_dir = os.path.join(args.save_dir, date_str, current_target)
                raw_dir = os.path.join(base_dir, 'raw')
                os.makedirs(raw_dir, exist_ok=True)
                
                # タイムスタンプでユニークなファイル名を作成
                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
                filepath = os.path.join(raw_dir, filename)
                
                # 枠が描かれていない「生の画像(color_image)」のみ保存
                cv2.imwrite(filepath, color_image)
                print(f"📸 画像のみを保存しました [{current_target}]: {filepath}")

            # 'a'キーで画像とアノテーションの両方を保存（オートアノテーション）
            elif key == ord('a'):
                date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                base_dir = os.path.join(args.save_dir, date_str, current_target)
                annotated_dir = os.path.join(base_dir, 'annotated')
                os.makedirs(annotated_dir, exist_ok=True)
                
                base_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                img_filepath = os.path.join(annotated_dir, base_filename + ".jpg")
                txt_filepath = os.path.join(annotated_dir, base_filename + ".txt")
                
                # [重要] アノテーション用には枠が描かれていない「生の画像(color_image)」を保存します
                cv2.imwrite(img_filepath, color_image)
                print(f"📸 画像を保存しました [{current_target}]: {img_filepath}")

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
                
                print(f"📝 アノテーションを自動保存しました [{current_target}]: {txt_filepath}")

            # ターゲット変更キーの処理
            elif key in target_classes:
                current_target = target_classes[key]
                print(f"🎯 撮影対象を '{current_target}' に変更しました。")

            # 'q'キーで終了
            elif key == ord('q'):
                break

    finally:
        # 終了処理
        print("終了処理を行っています...")
        try:
            pipeline.stop()
        except Exception as e:
            print(f"Pipeline stop時の警告 (既に停止している可能性があります): {e}")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
