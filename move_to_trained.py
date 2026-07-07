#!/usr/bin/env python3
import os
import shutil

def move_files_to_trained(base_dir):
    moved_count = 0
    # raw/ と annotated/ にあるファイルを検索
    for root, dirs, files in os.walk(base_dir):
        # 既にtrainedフォルダ内にあるものはスキップ
        if 'trained' in root.split(os.sep):
            continue

        rel_path = os.path.relpath(root, base_dir)
        parts = rel_path.split(os.sep)

        # parts[0]: 日付 (例: 2026-07-03)
        # parts[1]: ターゲット名 (block_red など)
        # parts[2]: 'raw' または 'annotated'
        if len(parts) >= 3 and parts[2] in ['raw', 'annotated']:
            date_str = parts[0]
            target = parts[1]
            type_dir = parts[2]
            
            dest_dir = os.path.join(base_dir, date_str, target, 'trained', type_dir)
            os.makedirs(dest_dir, exist_ok=True)

            for file in files:
                if file.endswith(('.jpg', '.txt')):
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(dest_dir, file)
                    
                    shutil.move(src_path, dest_path)
                    moved_count += 1
    
    print(f"✅ {moved_count} 個のファイルを 'trained' フォルダへ退避しました。")
    return moved_count

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), 'yolo_assets', 'collected_images')
    if not os.path.exists(base_dir):
        print(f"エラー: {base_dir} が見つかりません。")
        exit(1)
        
    print("📦 撮影済み画像を trained フォルダへ移動します...")
    move_files_to_trained(base_dir)
