

import trafilatura
import os
import csv
import urllib.request

# Thu muc
base_dir = "datasets/audio"
mp3_dir = os.path.join(base_dir, "mp3")
os.makedirs(mp3_dir, exist_ok=True)

# Hai danh sach url_audio va url_baibao
with open('audio_links.txt', 'r', encoding='utf-8') as f:
    audio_links = [line.strip() for line in f if line.strip()]

with open('newspaper_links.txt', 'r', encoding='utf-8') as f:
    newspaper_links = [line.strip() for line in f if line.strip()]

def download_dantri_audio():
    csv_path = os.path.join(base_dir, "train.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['file_name', 'link', 'content'])

        for i in range(len(audio_links)):
            audio_url = audio_links[i]
            newspaper_url = newspaper_links[i]

            # Tải nội dung bài báo
            downloaded = trafilatura.fetch_url(newspaper_url)
            article_text = trafilatura.extract(downloaded)

            # Tải file âm thanh
            try:
                audio_filename = os.path.join(mp3_dir, f"full_{i+1}.mp3")
                urllib.request.urlretrieve(audio_url, audio_filename)
                print(f"Đã tải: {audio_filename}")
                writer.writerow([f"full_{i+1}", newspaper_url, article_text])

            except Exception as e:
                print(f"Lỗi khi tải {audio_url}: {e}")
                continue
        
download_dantri_audio()