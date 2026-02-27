import os
import tempfile
import trafilatura
import fitz 
from docx import Document
import whisper
import yt_dlp
import re
from youtube_transcript_api import YouTubeTranscriptApi

class Extractor:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)

    def extract_website(self, url: str) -> str:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        return trafilatura.extract(downloaded) or ""

    def extract_text_file(self, file_path: str) -> str:
        text_content = ""
        
        if file_path.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                for page in doc:
                    text_content += page.get_text()
                    
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs]
            text_content = "\n".join(paragraphs)
            
        return text_content

    def extract_mp3(self, file_path: str) -> str:
        result = self.model.transcribe(file_path)
        return result["text"].strip()

    def extract_mp4(self, file_path: str) -> str:
        result = self.model.transcribe(file_path)
        return result["text"].strip()

    def extract_youtube(self, url: str) -> str:
        # 1. Thử lấy phụ đề trực tiếp (Ưu tiên tốc độ)
        video_id = None
        # Trích xuất Video ID từ URL bằng Regex
        id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if id_match:
            video_id = id_match.group(1)

        if video_id:
            try:
                ytt_api = YouTubeTranscriptApi()
                fetched_transcript = ytt_api.fetch(video_id, languages=['vi', 'en'])

                full_text = ''
                # is iterable
                for snippet in fetched_transcript:
                    full_text += snippet.text.replace(">>", "") + ' '
                return full_text.strip()

            except Exception as e:
                print(f"--- Lỗi khi lấy phụ đề: {str(e)} ---")
                print("--- Đang chuyển sang phương án dự phòng (Tải Audio & Whisper)... ---")

        # 2. Phương án dự phòng: Tải audio và dùng Whisper (Logic cũ của bạn)
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': '%(id)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,   
            'ignoreerrors': True,         
            'retries': 10,               
            'fragment_retries': 10,       
            'socket_timeout': 60,         
            'source_address': '0.0.0.0', 
        }

        print("--- Đang tải dữ liệu Audio từ YouTube... ---")
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts['outtmpl'] = os.path.join(temp_dir, '%(id)s.%(ext)s')
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                audio_file = filename.rsplit(".", 1)[0] + ".mp3"

                print("--- Đang sử dụng Whisper để chuyển đổi âm thanh thành văn bản... ---")
                text_content = self.extract_mp3(audio_file)
                
        return text_content

if __name__ == "__main__":
    extractor = Extractor()
    print(extractor.extract_youtube("https://www.youtube.com/watch?v=F4v-BH2EwYM"))
    
  