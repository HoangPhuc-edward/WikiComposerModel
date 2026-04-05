from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from fastapi.middleware.cors import CORSMiddleware

# Import các module của bạn
from llm_engine import LLMManager
from wiki_composer import WikiComposer
from template_manager import ContentTemplate

app = FastAPI(title="WikiCrop AI Writing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép mọi tên miền truy cập (hoặc thay bằng ["http://localhost", "http://127.0.0.1"])
    allow_credentials=True,
    allow_methods=["*"], # Cho phép mọi phương thức (GET, POST, OPTIONS...)
    allow_headers=["*"], # Cho phép mọi loại Header
)
# --- MODEL DỮ LIỆU ---
class WriteRequest(BaseModel):
    session_id: str
    topic_name: str
    template_data: Optional[dict] = None # Sau này dùng để truyền template tùy chỉnh từ Wiki

# --- HÀM HỖ TRỢ ---
def load_llm_config():
    config = {}
    try:
        with open("LLM.txt", "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip()
        return config
    except Exception as e:
        return None

# Khởi tạo LLM dùng chung
cfg = load_llm_config()
if not cfg:
    raise RuntimeError("Không thể nạp cấu hình LLM.txt")

my_llm = LLMManager(
    provider=cfg.get("Provider"),
    base_url=cfg.get("Base URL"),
    api_key=cfg.get("API KEY"),
    model_name=cfg.get("Model Name")
)

# Template mặc định (Có thể mở rộng để nhận từ request)
DEFAULT_TEMPLATE = ContentTemplate(
    name="Template mặc định",
    description="Tạo bài viết nông nghiệp chi tiết",
    system_instruction="Bạn là một chuyên gia về nông nghiệp, hãy viết một bài wiki vừa cô đọng vừa đơn giản",
    structure=[
        {"title": "Giới thiệu chung", "description": "Tổng quan về chủ đề"},
        {"title": "Nguồn gốc", "description": "Lịch sử và nguồn gốc hình thành"},
        {"title": "Đặc điểm nổi bật", "description": "Các đặc tính quan trọng nhất"}
    ]
)

# --- ENDPOINTS ---

@app.post("/api/v1/write")
async def write_wiki_article(request: WriteRequest):
    """
    Endpoint gọi WikiComposer để tạo bài viết dựa trên session_id đã có dữ liệu trong VectorDB.
    """
    try:
        # 1. Khởi tạo Composer với session_id từ request
        composer = WikiComposer(
            session_id=request.session_id,
            name=request.topic_name,
            template=DEFAULT_TEMPLATE,
            llm=my_llm
        )

        # 2. Thực hiện viết bài (RAG logic nằm bên trong wiki_compose)
        output = composer.wiki_compose()
        
        # 3. Lưu trữ kết quả vào file (giống logic cũ của bạn)
        path = f"output/{request.session_id}"
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, f"{request.session_id}.txt"), "w", encoding="utf-8") as f:
            f.write(output["full_content"])
            
        with open(os.path.join(path, f"{request.session_id}_array.json"), "w", encoding="utf-8") as f:
            json.dump(output["array_content"], f, ensure_ascii=False, indent=4)
            
        # 4. Trả về kết quả cho Frontend (MediaWiki)
        return {
            "status": "success",
            "session_id": request.session_id,
            "topic": request.topic_name,
            "content": output["full_content"],
            "array_content": output["array_content"],
            "bibliography": output["array_bibliography"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo bài viết: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "online", "model": cfg.get("Model Name")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)