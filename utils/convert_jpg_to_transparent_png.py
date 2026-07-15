#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fg_dir = os.path.join(project_root, 'yolo_assets', 'synthetic_sources', 'foregrounds')
    
    if not os.path.exists(fg_dir):
        print(f"エラー: ディレクトリが見つかりません: {fg_dir}")
        sys.exit(1)
        
    files = [f for f in os.listdir(fg_dir) if f.lower().endswith('.jpg')]
    if not files:
        print("変換対象のJPGファイルが見つかりません。")
        return
        
    print(f"{len(files)} 枚のJPG画像を透過PNGに変換します...")
    converted_count = 0
    
    for f in files:
        img_path = os.path.join(fg_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 画像の読み込みに失敗しました: {f}")
            continue
            
        h, w = img.shape[:2]
        
        # BGR画像のまま処理を行うため、RGBAへの変換は不要
        # floodFill 用のマスク (画像サイズより縦横2ピクセル大きい必要がある)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # 四隅の座標
        corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        
        # 背景透過の塗りつぶし実行
        # loDiff/upDiff で白（255, 255, 255）付近の微妙なブレ（240〜255）も透過対象にする
        # FLOODFILL_MASK_ONLY を指定し、入力画像は書き換えずに mask のみ 255 に塗りつぶす (255 << 8)
        for cx, cy in corners:
            # 既にマスクされている領域（mask座標は+1のオフセットがある）ならスキップ
            if mask[cy + 1, cx + 1] == 255:
                continue
                
            cv2.floodFill(img, mask, (cx, cy), None,
                          loDiff=(20, 20, 20), upDiff=(20, 20, 20),
                          flags=4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
                          
        # 画像サイズのマスク部分を取り出し
        bg_mask = mask[1:h+1, 1:w+1]
        
        # アルファチャンネルの作成 (背景は0:透明、それ以外は255:不透明)
        alpha = np.ones((h, w), dtype=np.uint8) * 255
        alpha[bg_mask == 255] = 0
        
        # BGR と Alpha を結合して BGRA にする
        bgra = cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha])
                          
        # 出力先のPNGパス
        png_name = os.path.splitext(f)[0] + '.png'
        png_path = os.path.join(fg_dir, png_name)
        
        # 保存
        cv2.imwrite(png_path, bgra)
        print(f"変換成功: {f} -> {png_name}")
        
        # 元のJPGを削除
        os.remove(img_path)
        converted_count += 1
        
    print(f"--- 変換完了 (成功: {converted_count}枚) ---")

if __name__ == '__main__':
    main()
