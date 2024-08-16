import cv2
import os
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def is_image_ratio_valid(image_path, target_ratio, tolerance=0.1):
    img = cv2.imread(image_path)
    if img is None:
        return False
    height, width = img.shape[:2]
    image_ratio = width / height
    return abs(image_ratio - target_ratio) <= tolerance

def process_image(input_folder, image, frames_per_image, target_size):
    img_path = os.path.join(input_folder, image)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return [img] * frames_per_image if img is not None else []

def create_flashback_video(input_folder, output_file, frame_rate=10, display_time=1, use_parallel=False, is_portrait=False):
    # 画像ファイルのリストを取得し、ソート
    all_images = sorted([img for img in os.listdir(input_folder) if img.lower().endswith((".png", ".jpg", ".jpeg"))])

    if not all_images:
        print("エラー: 指定されたフォルダに画像が見つかりません。")
        return

    # 最初の画像を読み込んで、フレームサイズと比率を取得
    first_image = cv2.imread(os.path.join(input_folder, all_images[0]))
    height, width = first_image.shape[:2]
    target_ratio = width / height

    # 有効な画像のみをフィルタリング
    images = [img for img in all_images if is_image_ratio_valid(os.path.join(input_folder, img), target_ratio)]

    if not images:
        print("エラー: 最初の画像と同じ比率の画像が見つかりません。")
        return

    # 目標サイズを設定（最大幅または高さを1920ピクセルに）
    if is_portrait:
        target_size = (int(1920 * target_ratio), 1920) if target_ratio < 1 else (1080, int(1080 / target_ratio))
    else:
        target_size = (1920, int(1920 / target_ratio)) if target_ratio > 1 else (int(1080 * target_ratio), 1080)

    # 動画ファイルの準備
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, frame_rate, target_size)

    # 1フレームあたりの表示フレーム数
    frames_per_image = int(frame_rate * display_time)

    # 画像の総数と動画の長さを計算
    total_images = len(images)
    video_duration = total_images * display_time

    # ユーザーに確認
    print(f"合計画像数: {total_images}")
    print(f"設定: {frame_rate} FPS, 画像表示時間: {display_time}秒")
    print(f"動画の長さ: {video_duration:.2f}秒")
    print(f"出力ファイル: {output_file}")
    print(f"並列処理: {'有効' if use_parallel else '無効'}")
    print(f"モード: {'縦長' if is_portrait else '横長'}")
    print(f"目標サイズ: {target_size}")

    if input("続行しますか？ (y/n): ").lower() != 'y':
        print("処理を中止しました。")
        return

    # プログレスバーの設定
    with tqdm(total=total_images, desc="画像処理中") as pbar:
        if use_parallel:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_image, input_folder, img, frames_per_image, target_size) for img in images]
                for future in as_completed(futures):
                    frames = future.result()
                    for frame in frames:
                        if frame is not None:
                            video.write(frame)
                    pbar.update(1)
        else:
            for img in images:
                frames = process_image(input_folder, img, frames_per_image, target_size)
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
    parser.add_argument('-o', '--output', type=str, default='output_video.mp4', help='Output video file path.')
    parser.add_argument('-f', '--fps', type=int, default=10, help='Frames per second for the output video.')
    parser.add_argument('-t', '--time', type=float, default=0.125, help='Display time for each image in seconds.')
    parser.add_argument('-p', '--parallel', action='store_true', help='Enable parallel processing.')
    parser.add_argument('-m', '--mode', type=str, choices=['portrait', 'landscape'], required=True, help='Video mode: portrait or landscape')

    args = parser.parse_args()

    # デフォルトの出力ファイル名に日時を追加
    if args.output == 'output_video.mp4':
        args.output = f"flashback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    create_flashback_video(args.input_folder, args.output, args.fps, args.time, args.parallel, args.mode == 'portrait')

if __name__ == "__main__":
    main()