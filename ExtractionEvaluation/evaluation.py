import os
import pandas as pd
import re
import csv
import time
from extractor import Extractor
import unicodedata
from thefuzz import fuzz

class Evaluator:
    def __init__(self, model_size="base"):
        self.extractor = Extractor(model_size=model_size)
        self.log_file = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, "w", newline='', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["METHOD", "IDENTIFIER", "RESULT", "DURATION_SECONDS", "SCORE", "THRESHOLD"])
            except Exception:
                pass

    def clean_text(self, text: str) -> str:
        if not text: return ""
        text = unicodedata.normalize('NFC', text.lower())
        text = re.sub(r'[^\w\s]', '', text)
        return " ".join(text.split())

    def compare(self, extracted: str, ground_truth: str, threshold=85):
        clean_extracted = self.clean_text(extracted)
        clean_truth = self.clean_text(ground_truth)

        if not clean_truth:
            return False, 0.0

        score = fuzz.ratio(clean_extracted, clean_truth)
        partial_score = fuzz.partial_ratio(clean_truth, clean_extracted)
        final_score = max(score, partial_score)

        if final_score >= threshold:
            return True, final_score
        else:
            print(f"--- Độ tương đồng thấp: {final_score}%")
            return False, final_score

    def log_result(self, method: str, identifier: str, result: str, duration: float = None, score: float = None, threshold: float = None):
        try:
            with open(self.log_file, "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([method, identifier, result, duration, score, threshold])
        except Exception as e:
            print(f"Lỗi khi ghi log: {e}")

    def run_audio_eval(self):
        print("\n>>> Đang chạy đánh giá: AUDIO (MP3)")
        csv_path = "datasets/audio/train.csv"
        audio_dir = "datasets/audio/mp3"
        
        try:
            df = pd.read_csv(csv_path)
            correct = 0
            
            for _, row in df.iterrows():
                file_path = os.path.join(audio_dir, f"{row['file_name']}.mp3")
                print(f"Đang xử lý file: {file_path}")
                
                start = time.time()
                try:
                    result = self.extractor.extract_mp3(file_path)
                    duration = round(time.time() - start, 3)
                    match, score = self.compare(result, row['content'])
                    if match:
                        correct += 1
                        print("Kết quả: ĐÚNG")
                        self.log_result("AUDIO", str(row.get('file_name', file_path)), "ĐÚNG", duration, score, 85)
                    else:
                        print("Kết quả: SAI")
                        self.log_result("AUDIO", str(row.get('file_name', file_path)), "SAI", duration, score, 85)
                except Exception as e:
                    duration = round(time.time() - start, 3)
                    print(f"Lỗi khi xử lý file {file_path}: {e}")
                    self.log_result("AUDIO", str(row.get('file_name', file_path)), "ERROR", duration, None, 85)
                    
            print(f"--- Tổng kết AUDIO: Đúng {correct}/{len(df)} dòng ---")
        except Exception as e:
            print(f"Lỗi đọc file CSV audio: {e}")

    def run_newspaper_eval(self):
        print("\n>>> Đang chạy đánh giá: NEWSPAPER (URL)")
        csv_path = "datasets/newspaper_link/train.csv"
        
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['X'])
            correct = 0
            
            for _, row in df.iterrows():
                url = row['X']
                print(f"Đang crawl URL: {url}")
                
                start = time.time()
                try:
                    result = self.extractor.extract_website(url)
                    duration = round(time.time() - start, 3)
                    match, score = self.compare(result, row['y'])
                    if match:
                        correct += 1
                        print("Kết quả: ĐÚNG")
                        self.log_result("NEWSPAPER", str(url), "ĐÚNG", duration, score, 85)
                    else:
                        print("Kết quả: SAI")
                        self.log_result("NEWSPAPER", str(url), "SAI", duration, score, 85)
                except Exception as e:
                    duration = round(time.time() - start, 3)
                    print(f"Lỗi khi crawl URL {url}: {e}")
                    self.log_result("NEWSPAPER", str(url), "ERROR", duration, None, 85)
                    
            print(f"--- Tổng kết NEWSPAPER: Đúng {correct}/{len(df)} dòng ---")
        except Exception as e:
            print(f"Lỗi đọc file CSV newspaper: {e}")

    def run_text_files_eval(self):
        print("\n>>> Đang chạy đánh giá: TEXT FILES (PDF/DOCX)")
        csv_path = "datasets/text_files/train.csv"
        file_dir = "datasets/text_files/file"
        
        try:
            df = pd.read_csv(csv_path)
            correct = 0
            
            for _, row in df.iterrows():
                file_name = row['file_name']
                possible_paths = [
                    os.path.join(file_dir, f"{file_name}.pdf"),
                    os.path.join(file_dir, f"{file_name}.docx"),
                    os.path.join(file_dir, file_name)
                ]
                
                target_path = next((p for p in possible_paths if os.path.exists(p)), None)
                
                if not target_path:
                    print(f"Không tìm thấy file: {file_name}")
                    continue
                    
                print(f"Đang đọc file: {target_path}")
                start = time.time()
                try:
                    result = self.extractor.extract_text_file(target_path)
                    duration = round(time.time() - start, 3)
                    match, score = self.compare(result, row['content'])
                    if match:
                        correct += 1
                        print("Kết quả: ĐÚNG")
                        self.log_result("TEXT FILES", str(file_name), "ĐÚNG", duration, score, 85)
                    else:
                        print("Kết quả: SAI")
                        self.log_result("TEXT FILES", str(file_name), "SAI", duration, score, 85)
                except Exception as e:
                    duration = round(time.time() - start, 3)
                    print(f"Lỗi xử lý file {target_path}: {e}")
                    self.log_result("TEXT FILES", str(file_name), "ERROR", duration, None, 85)
                    
            print(f"--- Tổng kết TEXT FILES: Đúng {correct}/{len(df)} dòng ---")
        except Exception as e:
            print(f"Lỗi đọc file CSV text_files: {e}")

    def run_video_eval(self):
        print("\n>>> Đang chạy đánh giá: VIDEO (MP4 & YOUTUBE)")
        csv_path = "datasets/video/train.csv"
        video_dir = "datasets/video/mp4"
        srt_dir = "datasets/video/subtitle"
        
        try:
            df = pd.read_csv(csv_path)
            correct = 0
            
            for _, row in df.iterrows():
                name = row['file_name']
                video_path = os.path.join(video_dir, f"{name}.mp4")
                srt_path = os.path.join(srt_dir, f"{name}.srt")
                yt_url = row['link']
                
                print(f"Đang kiểm tra cặp Video/Youtube: {name}")
                start = time.time()
                try:
                    ground_truth_content = ""
                    if os.path.exists(srt_path):
                        with open(srt_path, 'r', encoding='utf-8') as f:
                            ground_truth_content = f.read()
                    res_mp4 = self.extractor.extract_mp4(video_path)
                    res_yt = self.extractor.extract_youtube(yt_url)
                    duration = round(time.time() - start, 3)

                    match_mp4, score_mp4 = self.compare(res_mp4, ground_truth_content)
                    match_yt, score_yt = self.compare(res_yt, ground_truth_content)
                    match = match_mp4 or match_yt
                    score = max(score_mp4 or 0.0, score_yt or 0.0)

                    if match:
                        correct += 1
                        print(f"Kết quả {name}: ĐÚNG")
                        self.log_result("VIDEO", str(name), "ĐÚNG", duration, score, 85)
                    else:
                        print(f"Kết quả {name}: SAI")
                        self.log_result("VIDEO", str(name), "SAI", duration, score, 85)

                except Exception as e:
                    duration = round(time.time() - start, 3)
                    print(f"Lỗi xử lý video {name}: {e}")
                    self.log_result("VIDEO", str(name), "ERROR", duration, None, 85)
                    
            print(f"--- Tổng kết VIDEO: Đúng {correct}/{len(df)} dòng ---")
        except Exception as e:
            print(f"Lỗi đọc file CSV video: {e}")
    
    def clean_srt(self, srt_path: str) -> str:
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', content)
            content = re.sub(r'^\d+$', '', content, flags=re.MULTILINE)
            content = re.sub(r'<[^>]+>', '', content)
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            return " ".join(lines)
            
        except Exception as e:
            print(f"Lỗi đọc SRT {srt_path}: {e}")
            return ""
    
    def run_local_video_eval(self):
        print("\n>>> Đang chạy đánh giá: LOCAL VIDEO (MP4)")
        csv_path = "datasets/video/train.csv"
        video_dir = "datasets/video/mp4"
        srt_dir = "datasets/video/subtitle"
        
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['file_name'])
            correct = 0
            
            for _, row in df.iterrows():
                name = str(row['file_name'])
                video_path = os.path.join(video_dir, f"{name}.mp4")
                srt_path = os.path.join(srt_dir, f"{name}.srt")
                
                print(f"Đang kiểm tra Video Local: {name}")
                
                start = time.time()
                try:
                    ground_truth = ""
                    if os.path.exists(srt_path):
                        ground_truth = self.clean_srt(srt_path)
                    
                    if not ground_truth:
                        print(f"Bỏ qua {name}: Không tìm thấy hoặc lỗi file SRT")
                        continue

                    extracted_text = self.extractor.extract_mp4(video_path)
                    duration = round(time.time() - start, 3)

                    is_match, score = self.compare(extracted_text, ground_truth, threshold=75) # Video để threshold thấp hơn chút

                    if is_match:
                        correct += 1
                        print(f"Kết quả {name}: ĐÚNG (Score: {score}%)")
                        self.log_result("VIDEO_LOCAL", name, "ĐÚNG", duration, score, 75)
                    else:
                        print(f"Kết quả {name}: SAI (Score: {score}%)")
                        self.log_result("VIDEO_LOCAL", name, "SAI", duration, score, 75)

                except Exception as e:
                    duration = round(time.time() - start, 3)
                    print(f"Lỗi xử lý {name}: {e}")
                    self.log_result("VIDEO_LOCAL", name, "ERROR", duration, None, 75)
            
            print(f"--- Tổng kết LOCAL VIDEO: Đúng {correct}/{len(df)} ---")

        except Exception as e:
            print(f"Lỗi file CSV Video: {e}")

    def run_youtube_eval(self):
        print("\n>>> Đang chạy đánh giá: YOUTUBE EXTRACTOR")
        csv_path = "datasets/video/train.csv"
        srt_dir = "datasets/video/subtitle"
        
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['link'])
            correct = 0
            
            for _, row in df.iterrows():
                name = str(row['file_name'])
                link = row['link']
                srt_path = os.path.join(srt_dir, f"{name}.srt")
                
                print(f"Đang kiểm tra Youtube Link: {link}")
                
                start = time.time()
                try:
                    ground_truth = ""
                    if os.path.exists(srt_path):
                        ground_truth = self.clean_srt(srt_path)
                    
                    if not ground_truth:
                        print(f"Bỏ qua {name}: Cần file SRT local để đối chiếu kết quả Youtube")
                        continue

                    extracted_text = self.extractor.extract_youtube(link)
                    duration = round(time.time() - start, 3)
                    is_match, score = self.compare(extracted_text, ground_truth, threshold=75)

                    if is_match:
                        correct += 1
                        print(f"Kết quả {name}: ĐÚNG (Score: {score}%)")
                        self.log_result("YOUTUBE", name, "ĐÚNG", duration, score, 75)
                    else:
                        print(f"Kết quả {name}: SAI (Score: {score}%)")
                        self.log_result("YOUTUBE", name, "SAI", duration, score, 75)

                except Exception as e:
                    duration = round(time.time() - start, 3)
                    print(f"Lỗi xử lý Youtube {name}: {e}")
                    self.log_result("YOUTUBE", name, "ERROR", duration, None, 75)
            
            print(f"--- Tổng kết YOUTUBE: Đúng {correct}/{len(df)} ---")

        except Exception as e:
            print(f"Lỗi file CSV Video: {e}")

if __name__ == "__main__":
    evaluator = Evaluator(model_size="large-v3-turbo")
    # evaluator.run_audio_eval()
    # evaluator.run_newspaper_eval()
    # evaluator.run_text_files_eval()
    # evaluator.run_video_eval()
    # evaluator.run_local_video_eval()
    evaluator.run_youtube_eval()
