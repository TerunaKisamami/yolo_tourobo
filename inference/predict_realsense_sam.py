import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import argparse
import time
import torch
from ultralytics import YOLO, SAM
from yolo_detector import YoloDetector

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='RealSense YOLO+SAM Detection and Auto-Annotation')
    parser.add_argument('--save_dir', type=str, default='yolo_assets/collected_images', 
                        help='基本の保存ディレクトリパス (デフォルト: yolo_assets/collected_images)')
    parser.add_argument('--target', type=str, default='block_red',
                        help='初期の撮影対象名。実行中に0〜4キーで変更可能です。')
    parser.add_argument('--model_path', type=str, default='yolo_assets/robocon_models/custom_model_v1/weights/best.pt',
                        help='YOLOモデルのパス (デフォルト: yolo_assets/robocon_models/custom_model_v1/weights/best.pt)')
    parser.add_argument('--sam_model', type=str, default='sam2_b.pt',
                        help='SAMモデルのパス。sam2_b.pt, sam2_t.pt, sam_b.pt などが指定可能です (デフォルト: sam2_b.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='YOLOの検出しきい値 (デフォルト: 0.5)')
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

    # クラスごとのカラー設定 (BGR形式)
    class_colors = {
        'volleyball_pink': (180, 105, 255),  # 鮮やかなピンク
        'volleyball_cyan': (255, 255, 0),    # シアン
        'plate': (0, 165, 255),             # オレンジ
        'block_red': (0, 0, 255),           # 赤
        'block_blue': (255, 0, 0),          # 青
    }

    # 実行デバイスの判定 (CUDA GPUが利用可能ならGPUを使う)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ 処理デバイス: {device.upper()}")

    # 1. RealSenseのパイプライン初期化
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # カラーとデプスの位置合わせ用オブジェクト
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 2. YOLOとSAM検出器の初期化
    print(f"📦 YOLOモデル ({args.model_path}) をロード中...")
    detector = YoloDetector(model_path=args.model_path, conf_threshold=args.conf)
    
    print(f"📦 SAMモデル ({args.sam_model}) をロード中...")
    try:
        sam_model = SAM(args.sam_model)
    except Exception as e:
        print(f"⚠️ 指定されたSAMモデルの読み込みに失敗しました。デフォルトの 'sam2_t.pt' (軽量版) で再試行します。 エラー: {e}")
        sam_model = SAM('sam2_t.pt')

    # RealSense SDK 組み込みフィルターの初期化
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

    print("--------------------------------------------------")
    print("推論を開始します。終了するには画面を選択した状態で 'q' キーを押してください。")
    print("画像を保存するには 's' キーを押してください (rawのみ)。")
    print("画像とSAM補正アノテーションを保存するには 'a' キーを押してください (オートアノテーション)。")
    print("撮影対象の切り替え: 0: volleyball_pink, 1: volleyball_cyan, 2: plate, 3: block_red, 4: block_blue")
    print("--------------------------------------------------")

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

            # デプスフレームにSDKフィルターを適用
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)
            depth_frame = hole_filling_filter.process(depth_frame).as_depth_frame()

            # RealSenseフレームをOpenCV用numpy配列に変換
            color_image = np.asanyarray(color_frame.get_data())

            # 4. YOLOで物体検出
            # 表示用には生の画像をベースにするため、detector.detect の返り値(描画済画像)は使わずに
            # color_imageを複製して独自の描画を行います。
            _, detections = detector.detect(color_image)
            
            # 描画用のイメージを複製
            display_image = color_image.copy()

            # 5. SAMによるバウンディングボックスの精密化
            if len(detections) > 0:
                try:
                    # 全てのYOLO検出枠を抽出してリスト化
                    yolo_bboxes = [det['bbox'] for det in detections]
                    
                    # SAMに画像とバウンディングボックスのプロンプトを入力
                    sam_results = sam_model.predict(color_image, bboxes=yolo_bboxes, device=device, verbose=False)
                    
                    if sam_results and sam_results[0].masks is not None:
                        # セグメンテーションマスクの取得 (N, H, W)
                        masks = sam_results[0].masks.data.cpu().numpy()
                        
                        for i, det in enumerate(detections):
                            if i < len(masks):
                                mask = masks[i]
                                
                                # マスクのピクセルが存在する部分を検索
                                ys, xs = np.where(mask > 0)
                                if len(xs) > 0 and len(ys) > 0:
                                    # SAMが特定した精密な最小/最大の矩形を計算
                                    rx1 = float(np.min(xs))
                                    ry1 = float(np.min(ys))
                                    rx2 = float(np.max(xs))
                                    ry2 = float(np.max(ys))
                                    
                                    det['sam_bbox'] = [rx1, ry1, rx2, ry2]
                                    det['sam_active'] = True
                                    
                                    # [視覚効果] マスク部分を半透明でオーバーレイ描画
                                    cls_name = det.get('class_name', '')
                                    color = class_colors.get(cls_name, (0, 255, 0))
                                    color_mask = np.zeros_like(display_image)
                                    color_mask[mask > 0] = color
                                    display_image = cv2.addWeighted(display_image, 1.0, color_mask, 0.35, 0)
                                else:
                                    det['sam_bbox'] = det['bbox']
                                    det['sam_active'] = False
                            else:
                                det['sam_bbox'] = det['bbox']
                                det['sam_active'] = False
                except Exception as e:
                    print(f"SAM処理中にエラーが発生しました (YOLOの元枠にフォールバックします): {e}")
                    for det in detections:
                        det['sam_bbox'] = det['bbox']
                        det['sam_active'] = False
            
            # 6. 3D座標の計算と描画処理
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            for det in detections:
                cls_name = det.get('class_name', '')
                color = class_colors.get(cls_name, (0, 255, 0))
                
                # YOLO元の検出枠 (点線/グレーで薄く表示)
                yx1, yy1, yx2, yy2 = map(int, det['bbox'])
                cv2.rectangle(display_image, (yx1, yy1), (yx2, yy2), (120, 120, 120), 1, lineType=cv2.LINE_AA)
                
                # SAMで補正した枠を取得
                x1, y1, x2, y2 = map(int, det.get('sam_bbox', det['bbox']))
                is_sam = det.get('sam_active', False)
                
                # 補正後の枠を太い実線で描画
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                
                # 中心座標の計算
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cx = max(0, min(cx, color_frame.width - 1))
                cy = max(0, min(cy, color_frame.height - 1))
                
                # デプス取得モードの判定
                current_mode = depth_modes.get(cls_name, 'center')
                mode_closest = (current_mode == 'near')
                
                valid_distances = []
                if mode_closest:
                    # 最近点取得モード
                    margin_x = int((x2 - x1) * 0.15)
                    margin_y = int((y2 - y1) * 0.15)
                    search_x1 = max(0, x1 + margin_x)
                    search_x2 = min(color_frame.width - 1, x2 - margin_x)
                    search_y1 = max(0, y1 + margin_y)
                    search_y2 = min(color_frame.height - 1, y2 - margin_y)
                    
                    # 3ピクセル間隔で走査
                    for py in range(search_y1, search_y2 + 1, 3):
                        for px in range(search_x1, search_x2 + 1, 3):
                            dist = depth_frame.get_distance(px, py)
                            if dist > 0:
                                valid_distances.append(dist)
                else:
                    # 中心点取得モード (5x5のウィンドウ)
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
                        mode_text = "Near"
                    else:
                        distance = np.median(valid_distances)
                        mode_text = "Center"
                        
                    # 3D座標の計算
                    x_val, y_val, z_val = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], distance)
                    
                    # 画面表示用のテキスト
                    text_dist = f"Dist({mode_text}): {distance:.2f}m ({x_val*1000:.0f}, {y_val*1000:.0f}, {z_val*1000:.0f})mm"
                    
                    # 補正状態のテキスト
                    status_text = f"{cls_name} ({det['confidence']:.2f}) " + ("[SAM]" if is_sam else "[YOLO]")
                    
                    # 画面に情報を描画
                    cv2.circle(display_image, (cx, cy), 4, (0, 0, 255), -1)
                    
                    # テキスト表示（ラベルを上、距離を下）
                    cv2.putText(display_image, status_text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                    cv2.putText(display_image, text_dist, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # 時間計測とFPS計算
            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            fps = 1.0 / exec_time if exec_time > 0 else 0.0

            # 各種情報のオーバーレイ表示
            cv2.putText(display_image, f"Target: {current_target} (0-4 to change)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2, cv2.LINE_AA)
            cv2.putText(display_image, f"SAM Refinement Mode", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow('RealSense YOLO+SAM Detection', display_image)

            # キー入力の処理
            key = cv2.waitKey(1) & 0xFF
            
            # 's'キーで画像のみを保存 (raw)
            if key == ord('s'):
                date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                base_dir = os.path.join(args.save_dir, date_str, current_target)
                raw_dir = os.path.join(base_dir, 'raw')
                os.makedirs(raw_dir, exist_ok=True)
                
                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
                filepath = os.path.join(raw_dir, filename)
                
                # アノテーションを描画していない生の画像を保存
                cv2.imwrite(filepath, color_image)
                print(f"📸 生画像(raw)のみを保存しました [{current_target}]: {filepath}")

            # 'a'キーで画像とSAM補正アノテーションの両方を保存
            elif key == ord('a'):
                date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                base_dir = os.path.join(args.save_dir, date_str, current_target)
                annotated_dir = os.path.join(base_dir, 'annotated')
                os.makedirs(annotated_dir, exist_ok=True)
                
                base_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                img_filepath = os.path.join(annotated_dir, base_filename + ".jpg")
                txt_filepath = os.path.join(annotated_dir, base_filename + ".txt")
                
                # アノテーション用に生の画像を保存
                cv2.imwrite(img_filepath, color_image)
                print(f"📸 生画像を保存しました [{current_target}]: {img_filepath}")

                # アノテーション結果をYOLOフォーマットで保存 (SAMによる精密補正データを使用)
                img_h, img_w, _ = color_image.shape
                with open(txt_filepath, 'w') as f:
                    for det in detections:
                        x1, y1, x2, y2 = det.get('sam_bbox', det['bbox'])
                        cls_id = det['class_id']
                        
                        # YOLOフォーマットへの変換 (中心x, 中心y, 幅, 高さ を 0~1 に正規化)
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        bbox_w = (x2 - x1) / img_w
                        bbox_h = (y2 - y1) / img_h
                        
                        # クリッピング (0.0〜1.0の範囲外を防ぐ)
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        bbox_w = max(0.0, min(1.0, bbox_w))
                        bbox_h = max(0.0, min(1.0, bbox_h))
                        
                        # 保存
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")
                
                print(f"📝 SAM精密補正アノテーションを自動保存しました [{current_target}]: {txt_filepath}")

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
            print(f"Pipeline stop時の警告: {e}")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
