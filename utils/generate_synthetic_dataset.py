#!/usr/bin/env python3
import os
import sys
import random
import argparse
from PIL import Image

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.ANTIALIAS


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 合成データセット（コラージュ）生成スクリプト")
    parser.add_argument("--fg_dir", type=str, default="yolo_assets/synthetic_sources/foregrounds",
                        help="前景（透過PNG）画像ディレクトリのパス")
    parser.add_argument("--bg_dir", type=str, default="yolo_assets/synthetic_sources/backgrounds",
                        help="背景画像ディレクトリのパス")
    parser.add_argument("--output_dir", type=str, default="yolo_assets/datasets/synthetic_dataset",
                        help="生成したデータセットの出力ディレクトリ")
    parser.add_argument("--num_images", type=int, default=1000,
                        help="生成する画像枚数")
    parser.add_argument("--class_id", type=int, default=3,
                        help="合成するオブジェクトのクラスID（dataset.yamlに対応）")
    parser.add_argument("--min_scale", type=float, default=0.1,
                        help="背景に対する前景の最小スケール比率")
    parser.add_argument("--max_scale", type=float, default=0.3,
                        help="背景に対する前景の最大スケール比率")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="検証用データ (val) に割り振る割合 (0.0 ~ 1.0)")
    return parser.parse_args()

def generate_dataset():
    args = parse_args()

    # パスの解決
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fg_dir = os.path.join(project_root, args.fg_dir) if not os.path.isabs(args.fg_dir) else args.fg_dir
    bg_dir = os.path.join(project_root, args.bg_dir) if not os.path.isabs(args.bg_dir) else args.bg_dir
    output_dir = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    # 入力チェック
    if not os.path.exists(fg_dir):
        print(f"エラー: 前景画像ディレクトリが存在しません: {fg_dir}")
        print("背景透過したPNG画像（例: 箱単体の写真など）を配置してください。")
        sys.exit(1)
    if not os.path.exists(bg_dir):
        print(f"エラー: 背景画像ディレクトリが存在しません: {bg_dir}")
        print("合成に使用する背景用画像（例: 床やステージの写真など）を配置してください。")
        sys.exit(1)

    fg_files = [f for f in os.listdir(fg_dir) if f.lower().endswith('.png')]
    bg_files = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif'))]

    if not fg_files:
        print(f"エラー: 前景ディレクトリにPNG画像が見つかりません: {fg_dir}")
        sys.exit(1)
    if not bg_files:
        print(f"エラー: 背景ディレクトリに画像が見つかりません: {bg_dir}")
        sys.exit(1)

    # 出力先ディレクトリの構築 (YOLO標準構成: train/val 分割)
    subdirs = [
        "train/images", "train/labels",
        "val/images", "val/labels"
    ]
    for sd in subdirs:
        os.makedirs(os.path.join(output_dir, sd), exist_ok=True)

    print(f"--- 合成データセット作成開始 ---")
    print(f"前景画像数: {len(fg_files)}種類")
    print(f"背景画像数: {len(bg_files)}種類")
    print(f"生成総数: {args.num_images}枚 (Train: {int(args.num_images * (1 - args.val_split))}枚 / Val: {int(args.num_images * args.val_split)}枚)")
    print(f"対象クラスID: {args.class_id}")
    print(f"出力先: {output_dir}")

    success_count = 0
    
    for i in range(args.num_images):
        # データ分割の決定
        is_val = random.random() < args.val_split
        split_path = "val" if is_val else "train"

        # 前景と背景をランダムに選択
        fg_path = os.path.join(fg_dir, random.choice(fg_files))
        bg_path = os.path.join(bg_dir, random.choice(bg_files))

        try:
            fg_img = Image.open(fg_path).convert("RGBA")
            bg_img = Image.open(bg_path).convert("RGBA")
        except Exception as e:
            print(f"警告: 画像の読み込みに失敗したためスキップします ({fg_path} or {bg_path}): {e}")
            continue

        bg_w, bg_h = bg_img.size
        fg_w, fg_h = fg_img.size

        # 1. ランダムリサイズ (背景サイズに対するスケール比)
        scale = random.uniform(args.min_scale, args.max_scale)
        new_w = int(bg_w * scale)
        new_h = int(fg_h * (new_w / fg_w))
        if new_w <= 10 or new_h <= 10:
            continue
        
        try:
            fg_resized = fg_img.resize((new_w, new_h), LANCZOS)
        except Exception as e:
            print(f"警告: リサイズ中にエラーが発生しました: {e}")
            continue

        # 2. ランダム回転 (0 ~ 360度、expand=Trueでキャンバスを広げて端切れを防ぐ)
        angle = random.randint(0, 360)
        fg_rotated = fg_resized.rotate(angle, expand=True)

        # 3. 回転で出来た透明余白を取り除くためのバウンディングボックスの計算
        # getbbox() は透過アルファ値が0でない領域（＝実体領域）の (left, upper, right, lower) を返す
        bbox = fg_rotated.getbbox()
        if not bbox:
            # 万が一透過部分しかなかった場合はスキップ
            continue
        
        left, upper, right, lower = bbox
        r_w = right - left
        r_h = lower - upper

        # 4. 背景画像に配置可能なランダム位置の決定
        # 貼り付け位置は、実体領域が背景からはみ出さないように制限する
        max_x = bg_w - r_w
        max_y = bg_h - r_h

        if max_x <= 0 or max_y <= 0:
            # 背景に対して箱が大きすぎる場合はスキップ
            continue

        # 貼り付け位置 (x, y) は、bboxの左上(left, upper)が背景内のどこに来るか
        # 実際の貼り付け座標 paste_x, paste_y を計算
        actual_x = random.randint(0, max_x)
        actual_y = random.randint(0, max_y)
        paste_x = actual_x - left
        paste_y = actual_y - upper

        # 5. 合成
        combined = bg_img.copy()
        combined.alpha_composite(fg_rotated, (paste_x, paste_y))

        # 6. YOLOフォーマットのアノテーション座標計算
        # YOLOのBBox座標は [0, 1] に正規化された [中心x, 中心y, 幅, 高さ]
        cx = actual_x + (r_w / 2)
        cy = actual_y + (r_h / 2)

        yolo_cx = cx / bg_w
        yolo_cy = cy / bg_h
        yolo_w = r_w / bg_w
        yolo_h = r_h / bg_h

        # 座標が正常な範囲に収まっているか安全策
        yolo_cx = max(0.0, min(1.0, yolo_cx))
        yolo_cy = max(0.0, min(1.0, yolo_cy))
        yolo_w = max(0.0, min(1.0, yolo_w))
        yolo_h = max(0.0, min(1.0, yolo_h))

        # 7. 画像とラベルファイルの保存
        # JPEG保存のためにRGBに変換
        final_img = combined.convert("RGB")
        
        img_name = f"synth_{success_count:06d}.jpg"
        txt_name = f"synth_{success_count:06d}.txt"

        img_out_path = os.path.join(output_dir, split_path, "images", img_name)
        txt_out_path = os.path.join(output_dir, split_path, "labels", txt_name)

        try:
            final_img.save(img_out_path, "JPEG", quality=90)
            with open(txt_out_path, "w") as f:
                f.write(f"{args.class_id} {yolo_cx:.6f} {yolo_cy:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")
            success_count += 1
        except Exception as e:
            print(f"警告: 保存中にエラーが発生しました: {e}")
            continue

        # 進行状況表示
        if (i + 1) % 100 == 0 or (i + 1) == args.num_images:
            print(f"進行状況: {i + 1}/{args.num_images} 枚処理完了 (成功: {success_count} 枚)")

    print(f"\n--- 生成完了 ---")
    print(f"正常に生成された画像枚数: {success_count} 枚")
    print(f"データセットは {output_dir} に出力されました。")

if __name__ == "__main__":
    generate_dataset()
