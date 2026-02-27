import os
import json
import random
import pandas as pd
import gc # Giải phóng bộ nhớ
from datetime import datetime
from sentence_transformers import CrossEncoder

# Import modules
from preprocessor import Preprocessor
from wiki_composer import WikiComposer
from llm_engine import LLMManager
from template_manager import ContentTemplate
from wiki_evaluation import WikiEvaluation
from wiki_correctness import WikiCorrectness
import random
import string
# --- CẤU HÌNH ---
LLM_CONFIG_FILE = "LLM.txt"
DATA_SOURCE_FILE = "cleaned_data.json"
BASE_STORAGE = "data_storage"
RAW_SOURCE_DIR = os.path.join(BASE_STORAGE, "raw")

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# BASE_SESSION_ID = randomword(4) + "_"
BASE_SESSION_ID = "jwlk_"

def load_llm_config():
    """Nạp config LLM từ file"""
    config = {}
    try:
        with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip()
        return config
    except Exception as e:
        print(f"Lỗi nạp config: {e}"); return None

def step_1_generate_articles():
    """BƯỚC 1: VIẾT BÀI VÀ LƯU TRỮ (Tốn RAM cho Embedding/LLM)"""
    cfg = load_llm_config()
    if not cfg: return
    llm = LLMManager(provider=cfg.get("Provider"), base_url=cfg.get("Base URL"), 
                     api_key=cfg.get("API KEY"), model_name=cfg.get("Model Name"))

    with open(DATA_SOURCE_FILE, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"--- BẮT ĐẦU BƯỚC 1: SINH NỘI DUNG ({len(data_list)} bài) ---")
    for idx, item in enumerate(data_list):
        stt = BASE_SESSION_ID + str(idx + 1)
        # Tạo session_id cố định theo STT để Bước 2 dễ tìm kiếm
        session_id = f"article_{stt}" 
        print(f"\n🚀 Đang xử lý bài {stt}: {item.get('url')}")

        # 1. Nạp nguồn (ETL)
        preprocessor = Preprocessor(session_id=session_id)
        for src in item.get('source', []):
            if src.get('type') == "pdf":
                preprocessor.execute(src.get("url"), "pdf")
            elif src.get('type') == "txt":
                preprocessor.execute(src.get("url"), "txt")
            elif src.get('type') == "youtube":
                preprocessor.execute(src.get("url"), "youtube")
            else:
                preprocessor.execute(src.get("url"), "url")
        del preprocessor # Giải phóng preprocessor ngay sau khi xong

        # 2. Viết bài
        template = ContentTemplate(
            name=stt, description=f"wiki_của_{stt}",
            system_instruction="Bạn là chuyên gia nông nghiệp và biên tập viên Wiki",
            structure=item.get('structure', [])
        )
        composer = WikiComposer(session_id=session_id, name=item.get('name', "")+item.get("description", ""), template=template, llm=llm)
        output = composer.wiki_compose()

        # 3. Lưu trữ kết quả vào thư mục session
        path = f"output/{session_id}"
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{session_id}.txt"), "w", encoding="utf-8") as f: f.write(output["full_content"])
        with open(os.path.join(path, f"{session_id}_array.json"), "w", encoding="utf-8") as f: json.dump(output["array_content"], f, ensure_ascii=False, indent=4)
        with open(os.path.join(path, f"{session_id}_bibliography.json"), "w", encoding="utf-8") as f: json.dump(output["array_bibliography"], f, ensure_ascii=False, indent=4)
        
        print(f"✅ Đã lưu dữ liệu bài {stt} vào {path}")
        gc.collect() # Dọn rác bộ nhớ


def step_2_evaluate_and_csv():
    print("\n--- BẮT ĐẦU BƯỚC 2: ĐÁNH GIÁ ---")
    cfg = load_llm_config()
    max = 15

    llm = LLMManager(provider=cfg.get("Provider"), base_url=cfg.get("Base URL"), 
                     api_key=cfg.get("API KEY"), model_name=cfg.get("Model Name"))

    # Nạp Model NLI duy nhất 1 lần ở đây
    print("--- Đang nạp Model NLI dùng chung cho toàn bộ bài viết... ---")
    shared_nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    
    with open(DATA_SOURCE_FILE, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    results = []
    for idx, item in enumerate(data_list):
        if idx >= max:
            print(f"⚠️ Đã đạt giới hạn tối đa {max} bài, dừng chấm điểm.")
            break
        stt = BASE_SESSION_ID + str(idx + 1)
        session_id = f"article_{stt}"
        path = f"output/{session_id}"
        
        print(f"🔎 Đang chấm điểm bài {stt}...")
        try:
            with open(os.path.join(path, f"{session_id}_array.json"), 'r', encoding='utf-8') as f: ai_array = json.load(f)
            with open(os.path.join(path, f"{session_id}_bibliography.json"), 'r', encoding='utf-8') as f: ai_bib = json.load(f)
            with open(os.path.join(path, f"{session_id}.txt"), 'r', encoding='utf-8') as f: ai_text = f.read()

            # [YÊU CẦU]: Ghi lại bài viết vào log để kiểm tra
            with open(os.path.join(path, "evaluation_log.txt"), "w", encoding="utf-8") as f_log:
                f_log.write(f"--- BÀI VIẾT CỦA AI (SESSION: {session_id}) ---\n")
                f_log.write(ai_text)
                f_log.write("\n\n--- KẾT QUẢ ĐÁNH GIÁ SẼ ĐƯỢC CẬP NHẬT DƯỚI ĐÂY ---")

            # Đánh giá Citation - Truyền shared_nli_model vào
            evaluator = WikiEvaluation(ai_array, RAW_SOURCE_DIR, session_id, ai_bib, nli_model=shared_nli_model)
            evaluator.preprocess()
            c_recall = evaluator.calculate_avg_citation_recall()
            c_precision = evaluator.calculate_avg_citation_precision()

            evaluator.debug(os.path.join(path, f"{session_id}_citation_debug.json"))
            # Đánh giá Correctness - Truyền shared_nli_model vào
            c_eval = WikiCorrectness(ai_text, item.get('fullContent', ""), llm, path, nli_model=shared_nli_model)
            claim_recall = c_eval.calculate_correctness()
            # claim_recall = 0

            results.append({
                "url": item.get('url'),
                "citation_recall": round(c_recall, 2),
                "citation_precision": round(c_precision, 2),
                "correctness": round(claim_recall * 100, 2)
            })

            # Đợi 1 phút giữa các bài để tránh quá tải LLM
            print("⏳ Đang chờ 60 giây để tránh quá tải LLM...")
            import time
            time.sleep(60)
            
            # Ghi thêm kết quả vào log của bài đó cho dễ đọc
            with open(os.path.join(path, "evaluation_log.txt"), "a", encoding="utf-8") as f_log:
                f_log.write(f"\nRecall: {c_recall} | Precision: {c_precision} | Correctness: {claim_recall}")

        except Exception as e: print(f"❌ Lỗi tại bài {stt}: {e}")

    # Ghi file CSV như cũ
    df = pd.DataFrame(results)
    csv_file = f"output/final_evaluation_{datetime.now().strftime('%H%M%S')}.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"📊 Đã xuất báo cáo thành công: {csv_file}")

if __name__ == "__main__":
    # --- ĐIỀU KHIỂN QUY TRÌNH TẠI ĐÂY ---
    # BƯỚC 1: Chạy cái này trước, xong thì comment lại
    # step_1_generate_articles() 
    
    # BƯỚC 2: Chạy cái này sau khi Bước 1 đã hoàn tất
    step_2_evaluate_and_csv()