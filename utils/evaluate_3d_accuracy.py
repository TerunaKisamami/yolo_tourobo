#!/usr/bin/env python3
import os
import sys
import time
import argparse
import json
import datetime
import numpy as np
import cv2
import pyrealsense2 as rs

# プロジェクトルートディレクトリの設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# inference ディレクトリを sys.path に追加して YoloDetector をインポートできるようにする
inference_dir = os.path.join(project_root, 'inference')
if inference_dir not in sys.path:
    sys.path.append(inference_dir)

try:
    from yolo_detector import YoloDetector
except ImportError as e:
    print(f"警告: YoloDetectorのインポートに失敗しました。{e}")
    # テスト環境用にプレースホルダーを定義
    class YoloDetector:
        def __init__(self, *args, **kwargs): pass
        def detect(self, img): return img, []

def calculate_and_save_stats(samples, x_true, y_true, z_true, target_name, mode, output_path):
    """サンプリング結果から統計情報を計算し、コンソール出力およびJSONファイル保存を行います。"""
    # numpy 配列に変換 (N, 3)
    samples = np.array(samples)

    # 3D座標
    x_samples = samples[:, 0]
    y_samples = samples[:, 1]
    z_samples = samples[:, 2]

    # 実測距離 (各座標の原点からの距離)
    dist_samples = np.sqrt(x_samples**2 + y_samples**2 + z_samples**2)
    true_dist = np.sqrt(x_true**2 + y_true**2 + z_true**2)

    # 各軸の統計情報 (mm単位に変換)
    stats = {
        "evaluation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": target_name,
        "depth_mode": mode,
        "sample_count": len(samples),
        "ground_truth_mm": {
            "x": x_true * 1000,
            "y": y_true * 1000,
            "z": z_true * 1000,
            "distance": true_dist * 1000
        },
        "average_measured_mm": {
            "x": float(np.mean(x_samples) * 1000),
            "y": float(np.mean(y_samples) * 1000),
            "z": float(np.mean(z_samples) * 1000),
            "distance": float(np.mean(dist_samples) * 1000)
        },
        "std_dev_mm": {
            "x": float(np.std(x_samples) * 1000),
            "y": float(np.std(y_samples) * 1000),
            "z": float(np.std(z_samples) * 1000),
            "distance": float(np.std(dist_samples) * 1000)
        },
        "absolute_error_mm": {
            # 平均絶対誤差 (MAE)
            "x_mae": float(np.mean(np.abs(x_samples - x_true)) * 1000),
            "y_mae": float(np.mean(np.abs(y_samples - y_true)) * 1000),
            "z_mae": float(np.mean(np.abs(z_samples - z_true)) * 1000),
            "distance_mae": float(np.mean(np.abs(dist_samples - true_dist)) * 1000),
            # 最大誤差
            "x_max_err": float(np.max(np.abs(x_samples - x_true)) * 1000),
            "y_max_err": float(np.max(np.abs(y_samples - y_true)) * 1000),
            "z_max_err": float(np.max(np.abs(z_samples - z_true)) * 1000),
            "distance_max_err": float(np.max(np.abs(dist_samples - true_dist)) * 1000),
        }
    }

    # コンソール出力
    print("\n" + "="*65)
    print("📊 精度評価 統計結果 (単位: mm)")
    print("="*65)
    print(f"ターゲットクラス : {target_name} ({mode} mode)")
    print(f"サンプルフレーム数: {len(samples)}")
    print("-"*65)
    print("         |  実測値 (True) |  平均測定値  |  標準偏差  |  平均絶対誤差 (MAE)")
    print(f" X (横)  |   {stats['ground_truth_mm']['x']:11.1f}  |  {stats['average_measured_mm']['x']:11.1f}  |  {stats['std_dev_mm']['x']:9.2f}  |  {stats['absolute_error_mm']['x_mae']:11.1f}")
    print(f" Y (縦)  |   {stats['ground_truth_mm']['y']:11.1f}  |  {stats['average_measured_mm']['y']:11.1f}  |  {stats['std_dev_mm']['y']:9.2f}  |  {stats['absolute_error_mm']['y_mae']:11.1f}")
    print(f" Z (奥)  |   {stats['ground_truth_mm']['z']:11.1f}  |  {stats['average_measured_mm']['z']:11.1f}  |  {stats['std_dev_mm']['z']:9.2f}  |  {stats['absolute_error_mm']['z_mae']:11.1f}")
    print("-"*65)
    print(f"距離(3D) |   {stats['ground_truth_mm']['distance']:11.1f}  |  {stats['average_measured_mm']['distance']:11.1f}  |  {stats['std_dev_mm']['distance']:9.2f}  |  {stats['absolute_error_mm']['distance_mae']:11.1f}")
    print("-"*65)
    print(f"最大誤差: X={stats['absolute_error_mm']['x_max_err']:.1f}mm, Y={stats['absolute_error_mm']['y_max_err']:.1f}mm, Z={stats['absolute_error_mm']['z_max_err']:.1f}mm, 距離={stats['absolute_error_mm']['distance_max_err']:.1f}mm")
    print("="*65 + "\n")

    # ファイル出力
    if not output_path:
        output_path = os.path.join(project_root, "utils", "3d_accuracy_evaluation.json")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"💾 結果をファイルに保存しました: {output_path}")
    except Exception as e:
        print(f"❌ ファイル保存に失敗しました: {e}")

def main():
    parser = argparse.ArgumentParser(description='RealSense YOLO 3D Coordinate Accuracy Evaluation')
    parser.add_argument('--target', type=str, default='block_red',
                        choices=['volleyball_pink', 'volleyball_cyan', 'plate', 'block_red', 'block_blue'],
                        help='評価対象のターゲットクラス名')
    parser.add_argument('--x_true', type=float, required=True, help='実測のX座標 (mm)')
    parser.add_argument('--y_true', type=float, required=True, help='実測のY座標 (mm)')
    parser.add_argument('--z_true', type=float, required=True, help='実測のZ座標 (mm)')
    parser.add_argument('--frames', type=int, default=100, help='サンプリングする有効フレーム数')
    parser.add_argument('--output', type=str, default='', help='統計結果を保存するJSONパス')
    parser.add_argument('--mock', action='store_true', help='カメラ非接続時にシミュレーションデータで動作確認を行うテスト用モード')
    args = parser.parse_args()

    # 真値の座標 (メートル換算)
    x_true_m = args.x_true / 1000.0
    y_true_m = args.y_true / 1000.0
    z_true_m = args.z_true / 1000.0
    true_dist = np.sqrt(x_true_m**2 + y_true_m**2 + z_true_m**2)

    depth_modes = {
        'volleyball_pink': 'near',
        'volleyball_cyan': 'center',
        'plate': 'near',
        'block_red': 'center',
        'block_blue': 'center',
    }
    
    current_mode = depth_modes.get(args.target, 'center')

    # YOLO検出器の初期化
    model_path = os.path.join(project_root, 'yolo_assets/robocon_models/custom_model_v1/weights/best.pt')
    if not os.path.exists(model_path):
        print(f"警告: カスタムモデルが見つかりません。デフォルトの yolo11n.pt を使用します。")
        model_path = os.path.join(project_root, 'yolo11n.pt')
        
    try:
        detector = YoloDetector(model_path=model_path, conf_threshold=0.5)
    except Exception as e:
        print(f"YOLO検出器の初期化に失敗しました: {e}")
        return

    # モック動作モード (テスト環境用)
    if args.mock:
        print("🧪 [Mock Mode] シミュレーションで評価を行います...")
        mock_samples = []
        # 平均的に真値に近く、少量のノイズがのった座標を生成
        for _ in range(args.frames):
            mx = x_true_m + np.random.normal(0, 0.005) # 5mm程度の標準偏差
            my = y_true_m + np.random.normal(0, 0.005)
            mz = z_true_m + np.random.normal(0, 0.010) # 奥行きは少し多めのノイズ
            mock_samples.append((mx, my, mz))
        calculate_and_save_stats(mock_samples, x_true_m, y_true_m, z_true_m, args.target, current_mode, args.output)
        return

    # RealSense初期化
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # フィルター設定
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    hole_filling_filter = rs.hole_filling_filter()

    try:
        print("RealSenseを起動しています...")
        pipeline.start(config)
    except Exception as e:
        print(f"❌ RealSenseの起動に失敗しました: {e}")
        print("※実機カメラがない環境でテストしたい場合は --mock オプションを追加して起動してください。")
        return

    print("====================================================")
    print("🤖 YOLO 3D座標 精度評価ツール")
    print(f"  ターゲット: {args.target} (モード: {current_mode})")
    print(f"  実測座標 (真値): X={args.x_true:.1f}mm, Y={args.y_true:.1f}mm, Z={args.z_true:.1f}mm")
    print(f"  実測距離 (真値): {true_dist*1000:.1f}mm")
    print(f"  サンプリング目標: {args.frames} フレーム")
    print("====================================================")
    print("操作方法:")
    print("  'r' キー: 記録(サンプリング)開始")
    print("  'q' キー: 終了")
    print("====================================================")

    recording = False
    samples = []

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # フィルター適用
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)
            depth_frame = hole_filling_filter.process(depth_frame).as_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())

            # YOLOで物体検出
            annotated_image, detections = detector.detect(color_image)
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            # 指定されたターゲットのみを抽出
            target_dets = [d for d in detections if d.get('class_name') == args.target]

            detected_point = None

            if len(target_dets) > 0:
                # 複数検出された場合は、最もバウンディングボックスが大きいものを選択
                target_det = max(target_dets, key=lambda d: (d['bbox'][2]-d['bbox'][0]) * (d['bbox'][3]-d['bbox'][1]))
                x1, y1, x2, y2 = target_det['bbox']
                
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cx = max(0, min(cx, color_frame.width - 1))
                cy = max(0, min(cy, color_frame.height - 1))

                valid_distances = []
                mode_closest = (current_mode == 'near')

                if mode_closest:
                    margin_x = int((x2 - x1) * 0.15)
                    margin_y = int((y2 - y1) * 0.15)
                    search_x1 = max(0, int(x1) + margin_x)
                    search_x2 = min(color_frame.width - 1, int(x2) - margin_x)
                    search_y1 = max(0, int(y1) + margin_y)
                    search_y2 = min(color_frame.height - 1, int(y2) - margin_y)

                    for py in range(search_y1, search_y2 + 1, 3):
                        for px in range(search_x1, search_x2 + 1, 3):
                            dist = depth_frame.get_distance(px, py)
                            if dist > 0:
                                valid_distances.append(dist)
                else:
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
                        distance = np.percentile(valid_distances, 5)
                    else:
                        distance = np.median(valid_distances)

                    # 3D座標計算 (メートル)
                    x_val, y_val, z_val = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], distance)
                    detected_point = (x_val, y_val, z_val)

                    # リアルタイム表示用描画
                    color = (0, 165, 255) if mode_closest else (0, 255, 0)
                    cv2.circle(annotated_image, (cx, cy), 5, (0, 0, 255), -1)
                    text = f"Cur 3D: ({x_val*1000:.1f}, {y_val*1000:.1f}, {z_val*1000:.1f}) mm"
                    cv2.putText(annotated_image, text, (int(x1), int(y2) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if recording:
                        samples.append(detected_point)

            # 画面上に指示とステータスを表示
            status_text = "Status: READY (Press 'r' to start)"
            status_color = (0, 255, 255)
            if recording:
                status_text = f"Status: RECORDING... {len(samples)}/{args.frames}"
                status_color = (0, 165, 255)
                if len(samples) >= args.frames:
                    # 記録完了
                    recording = False
                    print(f"🎉 {args.frames}フレームのサンプリングが完了しました。統計を計算します...")
                    calculate_and_save_stats(samples, x_true_m, y_true_m, z_true_m, args.target, current_mode, args.output)
                    samples = [] # クリア
                    status_text = "Status: FINISHED (Results saved)"
                    status_color = (0, 255, 0)

            cv2.putText(annotated_image, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(annotated_image, f"True 3D: ({args.x_true:.1f}, {args.y_true:.1f}, {args.z_true:.1f}) mm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('YOLO 3D Accuracy Evaluation', annotated_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and not recording:
                print("サンプリングを開始します...")
                recording = True
                samples = []

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
