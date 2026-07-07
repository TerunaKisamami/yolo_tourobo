import os
import shutil
import re

base_dir = '/home/hatsu/Robobobo/yolo_tourobo/yolo_assets/collected_images'

moved_count = 0

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if not file.endswith(('.jpg', '.txt')):
            continue

        file_path = os.path.join(root, file)
        
        # 親フォルダが既に「YYYY-MM-DD」形式ならスキップ（仕分け済み）
        parent_dir_name = os.path.basename(root)
        if re.match(r'\d{4}-\d{2}-\d{2}', parent_dir_name):
            continue

        # ファイル名から日付を抽出 (例: 20260702_121818_759.jpg)
        match = re.match(r'(\d{4})(\d{2})(\d{2})_\d{6}_\d{3}', file)
        if match:
            year, month, day = match.groups()
            date_str = f"{year}-{month}-{day}"
        else:
            date_str = "unknown_date"

        # フォルダ構造からターゲット名と raw/annotated を判定
        rel_path = os.path.relpath(root, base_dir)
        parts = rel_path.split(os.sep)
        
        if len(parts) >= 2 and parts[1] in ['raw', 'annotated']:
            # 例: yolo_assets/collected_images/block_red/raw/xxx.jpg
            target = parts[0]
            type_dir = parts[1]
            dest_dir = os.path.join(base_dir, target, type_dir, date_str)
        else:
            # 旧バージョンで collected_images 直下に保存されたものなど
            # .txtがあるならannotated, なければrawと仮定
            if file.endswith('.txt'):
                type_dir = 'annotated'
            else:
                type_dir = 'raw'
            dest_dir = os.path.join(base_dir, 'uncategorized', type_dir, date_str)

        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, file)
        
        shutil.move(file_path, dest_path)
        moved_count += 1

print(f"仕分けが完了しました！ {moved_count} 個のファイルを移動しました。")
