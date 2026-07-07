#!/usr/bin/env python3
import os
import shutil
import subprocess

def merge_annotated_to_dataset(collected_dir, dataset_dir):
    print("🔄 新しいアノテーションデータをデータセットに統合します...")
    train_images_dir = os.path.join(dataset_dir, 'train', 'images')
    train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)

    copied_count = 0
    # collected_images/ 配下のファイルを探す
    for root, dirs, files in os.walk(collected_dir):
        # 既にtrainedに移動済みのものや、raw（未アノテーション）のものは除外
        if 'trained' in root.split(os.sep):
            continue

        if 'annotated' in root.split(os.sep):
            for file in files:
                src_path = os.path.join(root, file)
                if file.endswith('.jpg'):
                    dest_path = os.path.join(train_images_dir, file)
                    shutil.copy2(src_path, dest_path)
                    copied_count += 1
                elif file.endswith('.txt'):
                    dest_path = os.path.join(train_labels_dir, file)
                    shutil.copy2(src_path, dest_path)
                    copied_count += 1
    
    print(f"✅ {copied_count} 個のファイルをデータセットに追加しました。")
    return copied_count

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    collected_dir = os.path.join(base_dir, 'yolo_assets', 'collected_images')
    dataset_dir = os.path.join(base_dir, 'yolo_assets', 'datasets', 'robocon_data')

    print("="*50)
    print("🤖 完全ローカル自動再学習 (Auto-Retraining)")
    print("="*50)

    # 1. データの統合
    copied = merge_annotated_to_dataset(collected_dir, dataset_dir)
    if copied == 0:
        print("⚠️ 新しいアノテーションデータが見つかりませんでした。学習をスキップします。")
        exit(0)

    # 2. 学習の実行
    print("\n🚀 学習プロセス (train.py) を開始します...")
    result = subprocess.run(["python3", "train.py"], cwd=base_dir)
    if result.returncode != 0:
        print("❌ 学習中にエラーが発生しました。処理を中断します。")
        exit(1)

    # 3. OpenVINO変換
    print("\n🚀 OpenVINO形式へのコンバート (convert_openvino.py) を開始します...")
    result = subprocess.run(["python3", "convert_openvino.py"], cwd=base_dir)
    if result.returncode != 0:
        print("❌ 変換中にエラーが発生しました。処理を中断します。")
        exit(1)

    # 4. 退避処理
    print("\n📦 学習に使った画像を 'trained' フォルダへ退避します...")
    result = subprocess.run(["python3", "move_to_trained.py"], cwd=base_dir)
    if result.returncode == 0:
        print("\n🎉 全ての自動学習プロセスが完了しました！")
    else:
        print("\n❌ 退避処理中にエラーが発生しました。")
