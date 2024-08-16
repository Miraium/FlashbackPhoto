import cv2
import os
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def check_image(image_path, target_ratio, tolerance=0.1):
    img = cv2.imread(image_path)
    if img is None:
        return None
    height, width = img.shape[:2]
    image_ratio = width / height
    if abs(image_ratio - target_ratio) <= tolerance:
        return (image_path, width, height)
    return None

def find_valid_images(input_folder, is_portrait, tolerance=0.1):
    all_images = sorted([img for img in os.listdir(input_folder) if img.lower().endswith((".png", ".jpg", ".jpeg"))])
    
    # 最初の有効な画像を見つけて、その比率を基準にする
    first_valid_image = next((os.path.join(input_folder, img) for img in all_images if cv2.imread(os.path.join(input_folder, img)) is not None), None)
    if first_valid_image is None:
        return []
    
    first_img = cv2.imread(first_valid_image)
    height, width = first_img.shape[:2]
    target_ratio = width / height if not is_portrait else height / width
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(check_image, os.path.join(input_folder, img), target_ratio, tolerance): img for img in all_images}
        valid_images = []
        with tqdm(total=len(all_images), desc="画像チェック中") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    valid_images.append(result)
                pbar.update(1)
    
    # ソートされた順序を維持するためにソート
    valid_images.sort(key=lambda x: all_images.index(os.path.basename(x[0])))
    return valid_images

def calculate_target_size(width, height, is_portrait):
    if is_portrait:
        return (int(1920 * width / height), 1920) if height > width else (1080, int(1080 * height / width))
    else:
        return (1920, int(1920 * height / width)) if width > height else (int(1080 * width / height), 1080)

def process_image(input_folder, image, frames_per_image, target_size):
    img_path = os.path.join(input_folder, image)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return [img] * frames_per_image if img is not None else []

def create_flashback_video(input_folder, output_file, frame_rate=30, display_time=0.175, is_portrait=False, skip_confirmation=False):
    valid_images = find_valid_images(input_folder, is_portrait)

    if not valid_images:
        print("エラー: 指定されたモードに合う画像が見つかりません。")
        return

    # 最初の有効な画像からターゲットサイズを計算
    _, width, height = valid_images[0]
    target_size = calculate_target_size(width, height, is_portrait)

    # 画像の総数と動画の長さを計算
    total_images = len(valid_images)
    video_duration = total_images * display_time

    # 設定情報の表示
    print(f"合計画像数: {total_images}")
    print(f"設定: {frame_rate} FPS, 画像表示時間: {display_time}秒")
    print(f"動画の長さ: {video_duration:.2f}秒")
    print(f"出力ファイル: {output_file}")
    print(f"モード: {'縦長' if is_portrait else '横長'}")
    print(f"目標サイズ: {target_size}")

    if not skip_confirmation:
        if input("続行しますか？ (y/n): ").lower() != 'y':
            print("処理を中止しました。")
            return

    # 動画ファイルの準備
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, frame_rate, target_size)

    # 1フレームあたりの表示フレーム数
    frames_per_image = int(frame_rate * display_time)

    # プログレスバーの設定
    with tqdm(total=total_images, desc="画像処理中") as pbar:
        for img_path, _, _ in valid_images:
            frames = process_image(os.path.dirname(img_path), os.path.basename(img_path), frames_per_image, target_size)
            for frame in frames:
                if frame is not None:
                    video.write(frame)
            pbar.update(1)

    # リソースの解放
    video.release()

    # 動画ファイルが正しく作成されたか確認
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"動画ファイルが正常に作成されました: {output_file}")
    else:
        print(f"エラー: 動画ファイルの作成に失敗しました。")

def main():
    parser = argparse.ArgumentParser(description="Combine photos in chronological order into a flashback-style video.")
    parser.add_argument('input_folder', type=str, help='Folder containing input images.')
    parser.add_argument('-o', '--output', type=str, help='Output video file path.')
    parser.add_argument('-f', '--fps', type=int, default=30, help='Frames per second for the output video.')
    parser.add_argument('-t', '--time', type=float, default=0.175, help='Display time for each image in seconds.')
    parser.add_argument('-m', '--mode', type=str, choices=['portrait', 'landscape'], required=True, help='Video mode: portrait or landscape')
    parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation and proceed with video creation')

    args = parser.parse_args()

    # outputフォルダを作成（既に存在する場合は何もしない）
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    # 出力ファイルパスの設定
    if args.output is None:
        # 入力フォルダ名を取得
        input_folder_name = os.path.basename(os.path.normpath(args.input_folder))
        # デフォルトのファイル名を生成（flashback_入力フォルダ名_日時.mp4）
        default_filename = f"flashback_{input_folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        args.output = os.path.join(output_folder, default_filename)
    else:
        # 出力ファイルが指定されている場合、そのパスをそのまま使用
        args.output = args.output

    create_flashback_video(args.input_folder, args.output, args.fps, args.time, args.mode == 'portrait', args.yes)

if __name__ == "__main__":
    main()