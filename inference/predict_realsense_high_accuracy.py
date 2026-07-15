import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import time
import argparse
from collections import deque

# プロジェクトルートディレクトリの設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from inference.yolo_detector import YoloDetector

def main():
    parser = argparse.ArgumentParser(description='High Accuracy YOLO Detection')
    parser.add_argument('--target', type=str, default='block_red', help='Target class')
    args = parser.parse_args()

    current_target = args.target

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)

    # YOLO Detector
    model_path = os.path.join(project_root, 'yolo_assets/robocon_models/custom_model_v1/weights/best.pt')
    
    # 【精度特化設定1】 検出閾値を0.6に上げてブレた枠（確信度の低い枠）を排除
    detector = YoloDetector(model_path=model_path, conf_threshold=0.6)

    # 【精度特化設定2】 デプスフィルターの最大強化 (FPSとレイテンシを犠牲にしてツルツルにする)
    spatial_filter = rs.spatial_filter()
    spatial_filter.set_option(rs.option.filter_magnitude, 5)
    spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.25)
    spatial_filter.set_option(rs.option.filter_smooth_delta, 50)
    
    temporal_filter = rs.temporal_filter()
    temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.1) # 過去フレームへの依存度を高める
    temporal_filter.set_option(rs.option.filter_smooth_delta, 100)
    
    hole_filling_filter = rs.hole_filling_filter()

    # 【精度特化設定3】 移動平均バッファ (直近15フレーム分の X, Y, Z を保存)
    xyz_buffer = deque(maxlen=15)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"RealSenseの起動に失敗しました: {e}")
        return

    print("==================================================")
    print("【超高精度モード（テスト用）】推論を開始します。")
    print("注意: 座標の出力が安定するまでに約0.5秒の遅延（残像）が発生します。")
    print("終了するには画面を選択した状態で 'q' キーを押してください。")
    print("==================================================")

    try:
        prev_time = time.time()
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame: 
                continue

            # フィルターフル稼働 (ここがかなり重い)
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)
            depth_frame = hole_filling_filter.process(depth_frame).as_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())
            
            # 【精度特化設定4】 解像度を下げずに推論 (imgsz=640を明示)
            annotated_image, detections = detector.detect(color_image, imgsz=640)

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            
            # ターゲットのみをフィルタリング
            target_dets = [d for d in detections if d['class_name'] == current_target]
            
            if target_dets:
                # 複数検出された場合、最も信頼度の高い1つだけを追跡対象にする
                best_det = max(target_dets, key=lambda x: x['confidence'])
                x1, y1, x2, y2 = best_det['bbox']
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cx = max(0, min(cx, color_frame.width - 1))
                cy = max(0, min(cy, color_frame.height - 1))

                # 中心点取得モードで確実にデプスを取る (広めの 7x7 でスキャン)
                valid_distances = []
                half_w = 3
                for dy in range(-half_w, half_w + 1):
                    for dx in range(-half_w, half_w + 1):
                        px, py = cx + dx, cy + dy
                        if 0 <= px < color_frame.width and 0 <= py < color_frame.height:
                            dist = depth_frame.get_distance(px, py)
                            if dist > 0:
                                valid_distances.append(dist)
                
                if valid_distances:
                    distance = np.median(valid_distances)
                    x_val, y_val, z_val = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], distance)
                    
                    # 取得した XYZ (ミリメートル) をバッファに追加
                    xyz_buffer.append(np.array([x_val*1000, y_val*1000, z_val*1000]))
                    
                    # 移動平均の計算
                    avg_xyz = np.mean(xyz_buffer, axis=0)
                    avg_x, avg_y, avg_z = avg_xyz
                    
                    # 画面に平均化された座標を描画
                    text = f"Avg XYZ: ({avg_x:.0f}, {avg_y:.0f}, {avg_z:.0f})mm"
                    cv2.circle(annotated_image, (cx, cy), 4, (0, 0, 255), -1)
                    # 画面下部で見切れないように枠の上に文字を描画する
                    text_y = int(y1) - 10 if int(y1) - 10 > 20 else int(y2) + 20
                    cv2.putText(annotated_image, text, (int(x1), text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(annotated_image, "Depth Not Detected", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # ターゲットが見失われた場合はバッファをクリアし、過去の残像を断ち切る
                xyz_buffer.clear()

            # FPS計算
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            cv2.putText(annotated_image, f"High Accuracy Mode - Target: {current_target}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_image, f"FPS: {fps:.1f} (Buffer: {len(xyz_buffer)})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            cv2.imshow('RealSense High Accuracy Detection', annotated_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        print("終了処理を行っています...")
        try:
            pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
