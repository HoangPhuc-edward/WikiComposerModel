import os
import tempfile
import trafilatura
import fitz 
from docx import Document
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import re
import io

class Extractor:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self._whisper_model = None # Khởi tạo biến rỗng, không tải model ở đây!
        # Đã xóa dòng print gây hiểu lầm

    def _get_whisper_model(self):
        """Lazy Loading: Chỉ import và tải Whisper khi thực sự cần dùng đến audio"""
        if self._whisper_model is None:
            print(f"--- ⏳ Đang nạp mô hình Whisper ({self.model_size}) vào RAM... ---")
            # Import tại đây để tránh tải PyTorch vô ích khi chỉ đọc Web/PDF
            import whisper 
            self._whisper_model = whisper.load_model(self.model_size)
        return self._whisper_model

    def extract_website(self, url: str):
        print(f"🔍 [Website] Đang xử lý: {url}")
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                print(f"❌ [Website] Lỗi: Không thể tải nội dung tại {url}")
                return []
            
            text = trafilatura.extract(downloaded)
            if not text:
                print(f"⚠️ [Website] Cảnh báo: Trafilatura không trích xuất được văn bản từ {url}")
                return []
            
            print(f"✅ [Website] Trích xuất thành công ({len(text)} ký tự)")
            return [{
                "text": text,
                "metadata": {"source_type": "web", "locator": {}}
            }]
        except Exception as e:
            print(f"❌ [Website] Lỗi hệ thống khi xử lý {url}: {str(e)}")
            return []

    def extract_text_file(self, file_path: str):
        print(f"🔍 [File] Đang xử lý: {file_path}")
        final_result = []
        
        try:
            if file_path.endswith(".pdf"):
                with fitz.open(file_path) as doc:
                    for i, page in enumerate(doc, start=1):
                        text = page.get_text()
                        if text.strip():
                            final_result.append({
                                "text": text,
                                "metadata": {"source_type": "pdf", "locator": {"page_number": i}}
                            })
                
            elif file_path.endswith(".docx"):
                doc = Document(file_path)
                for i, para in enumerate(doc.paragraphs):
                    text = para.text.strip()
                    if text:
                        final_result.append({
                            "text": text,
                            "metadata": {"source_type": "docx", "locator": {"block_index": i}}
                        })
            else:
                print(f"❌ [File] Lỗi: Định dạng không hỗ trợ cho {file_path}")
                return []

            if not final_result:
                print(f"⚠️ [File] Cảnh báo: File {file_path} rỗng hoặc không có văn bản.")
            else:
                print(f"✅ [File] Trích xuất thành công {len(final_result)} đoạn.")
            
        except Exception as e:
            print(f"❌ [File] Lỗi khi đọc file {file_path}: {str(e)}")
    

    def extract_audio_content(self, file_content: bytes):
        """Trích xuất từ dữ liệu bytes của file Audio"""
        print(f"🔍 [Audio] Đang chuyển ngữ từ dữ liệu trực tiếp...")
        try:
            # Gọi model thông qua hàm lazy load, nó sẽ tự nhớ model cho các lần sau
            model = self._get_whisper_model()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            transcript = model.transcribe(tmp_path)
            os.remove(tmp_path) 

            final_result = []
            for segment in transcript["segments"]:
                text = segment["text"].strip()
                if text:
                    final_result.append({
                        "text": text,
                        "metadata": {
                            "source_type": "audio",
                            "locator": {"start_seconds": segment["start"], "end_seconds": segment["end"]}
                        }
                    })
            return final_result
        except Exception as e:
            print(f"❌ [Audio] Lỗi: {str(e)}")
            return []

    def extract_mp3(self, file_path: str):
        print(f"🔍 [Audio] Đang chuyển ngữ từ dữ liệu trực tiếp...")
        try:
            # Gọi model thông qua hàm lazy load, nó sẽ tự nhớ model cho các lần sau
            model = self._get_whisper_model()
            
            transcript = model.transcribe(file_path)
            os.remove(file_path) 

            final_result = []
            for segment in transcript["segments"]:
                text = segment["text"].strip()
                if text:
                    final_result.append({
                        "text": text,
                        "metadata": {
                            "source_type": "audio",
                            "locator": {"start_seconds": segment["start"], "end_seconds": segment["end"]}
                        }
                    })
            return final_result
        except Exception as e:
            print(f"❌ [Audio] Lỗi: {str(e)}")
            return []
       
    def extract_txt_content(self, file_content: bytes):
        """Trích xuất từ dữ liệu bytes của file .txt"""
        print(f"🔍 [Text] Đang xử lý nội dung văn bản trực tiếp...")
        final_result = []
        try:
            text_data = file_content.decode("utf-8")
            for line in text_data.splitlines():
                text = line.strip()
                if text:
                    final_result.append({
                        "text": text,
                        "metadata": {"source_type": "txt", "locator": {}}
                    })
            return final_result
        except Exception as e:
            print(f"❌ [Text] Lỗi: {str(e)}")
            return []

    def extract_youtube(self, url: str):
        print(f"🔍 [YouTube] Đang xử lý: {url}")
        video_id = None
        id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if id_match:
            video_id = id_match.group(1)

        if video_id:
            try:
                print(f"  -> Thử lấy phụ đề có sẵn cho {video_id}...")
                transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['vi', 'en'])
                
                final_result = []
                for snippet in transcript_list:
                    final_result.append({
                        "text": snippet.text,
                        "metadata": {
                            "source_type": "youtube",
                            "locator": {"start_seconds": snippet.start, "end_seconds": snippet.start + snippet.duration}
                        }
                    })
                print(f"✅ [YouTube] Lấy phụ đề thành công.")
                return final_result
            except Exception as e:
                print(f"  -> ⚠️ Phụ đề API thất bại: {str(e)}. Chuyển sang tải audio và transcribe...")

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '192'}],
            'quiet': True, 'no_warnings': True, 'nocheckcertificate': True, 'ignoreerrors': True,
        }

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts['outtmpl'] = os.path.join(temp_dir, '%(id)s.%(ext)s')
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if not info:
                        print(f"❌ [YouTube] Không thể tải video tại {url}")
                        return []
                    filename = ydl.prepare_filename(info)
                    audio_file = filename.rsplit(".", 1)[0] + ".mp3"
                    
                    # FIX LỖI: Đọc file mp3 thành bytes rồi đưa vào hàm extract_audio_content
                    with open(audio_file, "rb") as f:
                        audio_bytes = f.read()
                    
                    results = self.extract_audio_content(audio_bytes)
                    
                    for item in results: 
                        item['metadata']['source_type'] = "youtube"
                    return results
        except Exception as e:
            print(f"❌ [YouTube] Lỗi tải/xử lý audio cho {url}: {str(e)}")
            return []

    def extract_txt(self, file_path: str):
        print(f"🔍 [Text] Đang xử lý: {file_path}")
        final_result = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    text = line.strip()
                    if text:
                        final_result.append({
                            "text": text,
                            "metadata": {"source_type": "txt", "locator": {}}
                        })
            if not final_result:
                print(f"⚠️ [Text] Cảnh báo: File {file_path} rỗng hoặc không có văn bản.")
            else:
                print(f"✅ [Text] Trích xuất thành công {len(final_result)} dòng.")
        except Exception as e:
            print(f"❌ [Text] Lỗi khi đọc file {file_path}: {str(e)}")
        return final_result
