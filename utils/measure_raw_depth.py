import pyrealsense2 as rs
import numpy as np
import cv2
import collections
import os
import sys

# プロジェクトルートディレクトリの設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def main():
    # RealSense初期化
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # RealSense SDK 組み込みフィルターの初期化（デプス欠損ゼロ対策）
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    hole_filling_filter = rs.hole_filling_filter()

    try:
        print("RealSenseを起動しています...")
        profile = pipeline.start(config)
    except Exception as e:
        print(f"起動に失敗しました: {e}")
        return

    print("--- 距離測定ツール ---")
    print("画面中央の十字（赤）が指す対象との距離を測定します。")
    print("平らな壁に向けて静止させ、ブレ（標準偏差）などを確認してください。")
    print("終了するには 'q' キーを押してください。")
    print("精度評価を記録するには 'r' キーを押してください（100フレーム分記録します）。")

    # 過去30フレーム分の距離データを保存するキュー (約1秒分)
    history_len = 30
    dist_history = collections.deque(maxlen=history_len)

    # 評価用の変数
    recording = False
    record_data = []
    record_frames_target = 100

    try:
        while True:
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

            color_image = np.asanyarray(color_frame.get_data())

            # 画面中心の座標 (640x480 の中央)
            cx, cy = 320, 240

            # 中心点から5x5ピクセルの領域で0以外の距離を取得（デプス欠損対策）
            half_w = 2
            valid_distances = []
            for dy in range(-half_w, half_w + 1):
                for dx in range(-half_w, half_w + 1):
                    px, py = cx + dx, cy + dy
                    if 0 <= px < color_frame.width and 0 <= py < color_frame.height:
                        dist = depth_frame.get_distance(px, py)
                        if dist > 0:
                            valid_distances.append(dist)
            
            # 有効な距離データがあれば、その中央値を採用する
            distance = 0.0
            if len(valid_distances) > 0:
                distance = np.median(valid_distances)

            if distance > 0:
                dist_history.append(distance)
                if recording:
                    record_data.append(distance)
                    if len(record_data) >= record_frames_target:
                        recording = False
                        # 評価を計算してファイルとコンソールに出力
                        eval_avg = np.mean(record_data)
                        eval_std = np.std(record_data)
                        eval_min = np.min(record_data)
                        eval_max = np.max(record_data)
                        import json
                        out_data = {
                            "frames": record_frames_target,
                            "average_mm": eval_avg * 1000,
                            "std_dev_mm": eval_std * 1000,
                            "min_mm": eval_min * 1000,
                            "max_mm": eval_max * 1000,
                            "fluctuation_mm": (eval_max - eval_min) * 1000
                        }
                        with open(os.path.join(project_root, "depth_evaluation.json"), "w") as f:
                            json.dump(out_data, f, indent=4)
                        print(f"\n=== 精度評価結果 ({out_data['average_mm']:.1f}mm) ===")
                        print(f"Average: {out_data['average_mm']:.2f} mm")
                        print(f"StdDev : {out_data['std_dev_mm']:.2f} mm")
                        print(f"Min/Max: {out_data['min_mm']:.2f} / {out_data['max_mm']:.2f} mm")
                        print(f"Range  : {out_data['fluctuation_mm']:.2f} mm")
                        print("==================================\n")
            
            # 統計値の計算
            if len(dist_history) > 0:
                avg_dist = np.mean(dist_history)
                std_dist = np.std(dist_history)  # 標準偏差（ブレの大きさ）
                min_dist = np.min(dist_history)
                max_dist = np.max(dist_history)
                diff_dist = max_dist - min_dist  # 最大と最小の振幅
            else:
                avg_dist, std_dist, diff_dist = 0, 0, 0

            # 画面への描画
            # 中央にターゲット用の十字キーを描画
            cv2.drawMarker(color_image, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            # 測定データのテキストオーバーレイ
            # メートル表示とミリメートル表示
            cv2.putText(color_image, f"Current: {distance:.4f} m ({distance*1000:.1f} mm)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(color_image, f"Average: {avg_dist:.4f} m ({avg_dist*1000:.1f} mm)", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(color_image, f"Jitter (StdDev): {std_dist*1000:.2f} mm", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if std_dist*1000 > 10 else (0, 255, 0), 2)
            cv2.putText(color_image, f"Fluctuation range: {diff_dist*1000:.2f} mm", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(color_image, f"Data count: {len(dist_history)}/{history_len}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if recording:
                cv2.putText(color_image, f"Recording... {len(record_data)}/{record_frames_target}", (20, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow('RealSense Raw Depth Measure', color_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and not recording:
                print("記録を開始します...")
                recording = True
                record_data = []

    finally:
        print("終了処理を行っています...")
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
