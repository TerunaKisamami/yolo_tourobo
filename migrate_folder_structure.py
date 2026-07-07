#!/usr/bin/env python3
import os
import shutil
import re

def migrate_structure(base_dir):
    moved_count = 0
    # 一時的な退避先（同じディレクトリ内での移動による無限ループ防止）
    temp_dir = os.path.join(base_dir, '_temp_migration')
    os.makedirs(temp_dir, exist_ok=True)
    
    files_to_move = []

    # 1. すべてのファイルを収集し、移動先のパスを決定する
    for root, dirs, files in os.walk(base_dir):
        if '_temp_migration' in root:
            continue

        for file in files:
            if not file.endswith(('.jpg', '.txt')):
                continue

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, base_dir)
            parts = rel_path.split(os.sep)

            # すでに新しい階層 (Date -> Target -> ...) になっているかチェック
            if len(parts) >= 1 and re.match(r'\d{4}-\d{2}-\d{2}', parts[0]):
                continue # 新階層のファイルはスキップ

            # 日付の取得（ファイル名から）
            match = re.match(r'(\d{4})(\d{2})(\d{2})_\d{6}_\d{3}', file)
            if match:
                year, month, day = match.groups()
                date_str = f"{year}-{month}-{day}"
            else:
                date_str = "unknown_date"

            # ターゲット、trained、type (raw/annotated) の判定
            target = parts[0] if len(parts) > 0 else 'uncategorized'
            is_trained = 'trained' in parts
            
            if 'annotated' in parts:
                type_dir = 'annotated'
            elif 'raw' in parts:
                type_dir = 'raw'
            else:
                type_dir = 'annotated' if file.endswith('.txt') else 'raw'

            # 新しいディレクトリパスの構築
            # 新階層: Date / Target / [trained] / Type
            if is_trained:
                dest_dir = os.path.join(base_dir, date_str, target, 'trained', type_dir)
            else:
                dest_dir = os.path.join(base_dir, date_str, target, type_dir)

            dest_path = os.path.join(dest_dir, file)
            
            # 同じパスなら移動不要
            if file_path == dest_path:
                continue

            # 一旦テンポラリに移動する（名前衝突や上書きを防ぐため）
            temp_path = os.path.join(temp_dir, f"{moved_count}_{file}")
            shutil.move(file_path, temp_path)
            
            files_to_move.append((temp_path, dest_path))
            moved_count += 1

    # 2. テンポラリから本来の目的地へ移動
    for temp_path, dest_path in files_to_move:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(temp_path, dest_path)

    # 3. テンポラリと空になった古いフォルダの削除
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # 空ディレクトリの再帰的削除
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except OSError:
                pass

    print(f"📦 階層のマイグレーションが完了しました！ {moved_count} 個のファイルを新階層へ移動しました。")

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo_assets', 'collected_images')
    if os.path.exists(base_dir):
        migrate_structure(base_dir)
    else:
        print("エラー: collected_images フォルダが見つかりません。")
