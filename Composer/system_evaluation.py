import os
import json
import random
import pandas as pd
import gc # Giải phóng bộ nhớ
from datetime import datetime
from sentence_transformers import CrossEncoder
import time
# Import modules
from preprocessor import Preprocessor
from wiki_composer import WikiComposer
from llm_engine import LLMManager
from template_manager import ContentTemplate
from wiki_evaluation import WikiEvaluation
from wiki_correctness import WikiCorrectness
import random
import string


# Source data
DATA_SOURCE_FILE = "cleaned_data.json"
BASE_STORAGE = "data_storage"
RAW_SOURCE_DIR = os.path.join(BASE_STORAGE, "raw")

# Helper func
def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# BASE_SESSION_ID = randomword(4) + "_"
BASE_SESSION_ID = "test_"
def load_llm_config(LLM_CONFIG_FILE):
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
    
    cfg = load_llm_config("LLM/LLM.txt")
    cfg_small = load_llm_config("LLM/LLM_small.txt")
    if not cfg: return
    llm = LLMManager(provider=cfg.get("Provider"), base_url=cfg.get("Base URL"), 
                     api_key=cfg.get("API KEY"), model_name=cfg.get("Model Name"))
    llm_small = LLMManager(provider=cfg_small.get("Provider"), base_url=cfg_small.get("Base URL"), 
                           api_key=cfg_small.get("API KEY"), model_name=cfg_small.get("Model Name"))

    with open(DATA_SOURCE_FILE, 'r', encoding='utf-8') as f:
        data_list = json.load(f)


    start_idx = 17
    data_list = data_list[17:18]  
    print(f"WRITTING: ({len(data_list)} bài) ---")
    
    for idx, item in enumerate(data_list):
        stt = BASE_SESSION_ID + str(idx + 1 + start_idx)  
        
        # Create session_id
        session_id = f"article_{stt}" 
        print(f"\nĐang xử lý bài {stt}: {item.get('url')}")

        # 1. Source input
        preprocessor = Preprocessor(session_id=session_id)
        for src in item.get('source', []):
            if src.get('type') == "pdf":
                preprocessor.execute(src.get("url"), "pdf")
            elif src.get('type') == "txt":
                preprocessor.execute(src.get("url"), "txt")
            elif src.get('type') == "youtube":
                preprocessor.execute(src.get("url"), "youtube")
            elif src.get('type') == "audio":
                preprocessor.execute(src.get("url"), "audio")
            else:
                preprocessor.execute(src.get("url"), "url")
        del preprocessor 

        # 2. Write wiki
        template = ContentTemplate(
            name=stt, description=f"wiki_của_{stt}",
            system_instruction="Bạn là chuyên gia nông nghiệp và biên tập viên Wiki",
            structure=item.get('structure', [])
        )
        composer = WikiComposer(session_id=session_id, name=item.get('name', "")+item.get("description", ""), template=template, llm=llm, llm_small=llm_small)
        output = composer.wiki_compose()

        # 3. Save output
        path = f"output/{session_id}"
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{session_id}.txt"), "w", encoding="utf-8") as f: f.write(output["full_content"])
        with open(os.path.join(path, f"{session_id}_array.json"), "w", encoding="utf-8") as f: json.dump(output["array_content"], f, ensure_ascii=False, indent=4)
        with open(os.path.join(path, f"{session_id}_bibliography.json"), "w", encoding="utf-8") as f: json.dump(output["array_bibliography"], f, ensure_ascii=False, indent=4)
        
        print(f"Đã lưu dữ liệu bài {stt} vào {path}")
        gc.collect() 
        

def step_2_evaluate_and_csv():
    print("\nEVALUATION: ---")
    cfg = load_llm_config("LLM/LLM_test.txt")

    llm = LLMManager(provider=cfg.get("Provider"), base_url=cfg.get("Base URL"), 
                     api_key=cfg.get("API KEY"), model_name=cfg.get("Model Name"))

    # Nạp Model NLI 
    print("--- Đang nạp Model NLI dùng chung cho toàn bộ bài viết... ---")
    shared_nli_model = CrossEncoder('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')
    
    with open(DATA_SOURCE_FILE, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    start_idx = 0
    data_list = data_list
    results = []
    for idx, item in enumerate(data_list):
        
        stt = BASE_SESSION_ID + str(idx + 1 + start_idx)
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

            print("Đang chờ 20 giây để tránh quá tải LLM...")
            import time
            time.sleep(20)
            
            with open(os.path.join(path, "evaluation_log.txt"), "a", encoding="utf-8") as f_log:
                f_log.write(f"\nRecall: {c_recall} | Precision: {c_precision} | Correctness: {claim_recall}")

        except Exception as e: print(f"Lỗi tại bài {stt}: {e}")

    # Save CSV
    df = pd.DataFrame(results)
    csv_file = f"output/{BASE_SESSION_ID}_final_evaluation_{datetime.now().strftime('%H%M%S')}.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"Output evaluation: {csv_file}")

if __name__ == "__main__":
    # step_1_generate_articles() 
    
    step_2_evaluate_and_csv()