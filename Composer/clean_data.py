import os
import json
import re
from llm_engine import LLMManager
from extractor import Extractor

# --- CẤU HÌNH ---
INPUT_FILE = "uncleaned_data.jsonl"
OUTPUT_FILE = "cleaned_data.json"
LLM_CONFIG_FILE = "LLM.txt"

# Danh sách các model để xoay vòng khi lỗi/hết token
LLM_NAMES = [
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m"
]

class WikiDataPreprocessor:
    def __init__(self):
        self.current_model_idx = 0
        self.extractor = Extractor(model_size="base")
        self.cfg = self._load_llm_config()
        self.llm = self._init_llm()

    def _load_llm_config(self):
        config = {}
        try:
            with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        config[key.strip()] = value.strip()
            return config
        except Exception as e:
            print(f"❌ Lỗi nạp LLM config: {e}")
            return None

    def _init_llm(self):
        """Khởi tạo LLM với model hiện tại trong danh sách"""
        model_name = LLM_NAMES[self.current_model_idx]
        print(f"🤖 Đang sử dụng model: {model_name}")
        return LLMManager(
            provider=self.cfg.get("Provider"),
            base_url=self.cfg.get("Base URL"),
            api_key=self.cfg.get("API KEY"),
            model_name=model_name
        )

    def _call_llm_with_rotation(self, prompt):
        """Gọi LLM, nếu lỗi thì tự động đổi sang model tiếp theo"""
        attempts = 0
        while attempts < len(LLM_NAMES):
            try:
                response = self.llm.send_prompt(prompt, options={"temperature": 0.1})
                if response:
                    return response
            except Exception as e:
                print(f"⚠️ Model {LLM_NAMES[self.current_model_idx]} gặp lỗi: {e}")
                self.current_model_idx = (self.current_model_idx + 1) % len(LLM_NAMES)
                self.llm = self._init_llm() # Khởi tạo lại với model mới
                attempts += 1
        return None

    def _parse_llm_output(self, response):
        """Trích xuất JSON từ chuỗi phản hồi của LLM"""
        try:
            # Tìm đoạn JSON trong chuỗi (giả sử LLM trả về có bọc trong ```json)
            json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
            return json.loads(json_str)
        except:
            return None

    def process(self):
        if not self.cfg: return
        
        # Đọc toàn bộ dữ liệu đầu vào (JSONL)
        uncleaned_arr = []
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                uncleaned_arr.append(json.loads(line))

        print(f"📂 Bắt đầu xử lý {len(uncleaned_arr)} bài viết...")
        
        # Để lưu vào file JSON chuẩn, ta sẽ mở file và ghi dần theo dạng list
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
            f_out.write("[\n") # Bắt đầu mảng JSON

            for i, item in enumerate(uncleaned_arr):
                print(f"⏳ Processing [{i+1}/{len(uncleaned_arr)}]: {item.get('url')}")
                
                # 1. Xử lý qua LLM để lấy Name, Description và Structure
                extended_headings = ["Giới thiệu"] + item.get('heading', [])
            
                # 1. Xử lý qua LLM với danh sách heading đã được mở rộng
                prompt = f"""
                Dựa vào nội dung bài Wiki và danh sách đề mục sau, hãy trích xuất thông tin.
                Lưu ý: Đề mục "Giới thiệu" tương ứng với phần mở đầu của bài viết (trước tiêu đề đầu tiên).

                Nội dung: {item['fullContent']} 
                Danh sách đề mục cần tóm tắt: {extended_headings}

                Hãy trả về duy nhất một mã JSON theo cấu trúc sau:
                {{
                    "name": "Tên bài viết",
                    "description": "Mô tả tổng quát bài viết khoảng 50 chữ",
                    "structure": [
                        {{ "title": "Tên đề mục", "description": "Mô tả ngắn gọn nội dung đề mục này nói gì" }}
                    ]
                }}
                """
                
                llm_response = self._call_llm_with_rotation(prompt)
                parsed_data = self._parse_llm_output(llm_response) if llm_response else None

                if not parsed_data:
                    print(f"❌ Không thể phân tích dữ liệu bài {i+1}")
                    continue

                # 2. Xử lý Source bằng Extractor
                processed_sources = []
                for src_url in item.get('source', []):
                    # Kiểm tra xem có crawl được không
                    # Lưu ý: extractor.extract_website trả về list nếu thành công
                    is_crawlable = len(self.extractor.extract_website(src_url)) > 0
                    processed_sources.append({
                        "url": src_url,
                        "type": "web" if is_crawlable else "no"
                    })

                # 3. Tổng hợp kết quả
                result_item = {
                    "url": item['url'],
                    "name": parsed_data.get("name"),
                    "description": parsed_data.get("description"),
                    "fullContent": item['fullContent'],
                    "structure": parsed_data.get("structure"),
                    "source": processed_sources
                }

                # 4. Ghi trực tiếp vào file (Incremental Saving)
                json_str = json.dumps(result_item, ensure_ascii=False, indent=4)
                f_out.write(json_str)
                
                # Nếu chưa phải phần tử cuối, thêm dấu phẩy
                if i < len(uncleaned_arr) - 1:
                    f_out.write(",\n")
                else:
                    f_out.write("\n")
                
                # Flush để đảm bảo dữ liệu được ghi xuống đĩa ngay lập tức
                f_out.flush()

            f_out.write("]") # Kết thúc mảng JSON
        
        print(f"✅ Hoàn tất! Dữ liệu đã được lưu tại {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocessor = WikiDataPreprocessor()
    preprocessor.process()