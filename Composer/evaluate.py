import os
import json
import re
import sys
import random
import string
from preprocessor import Preprocessor
from wiki_composer import WikiComposer
from llm_engine import LLMManager
from template_manager import ContentTemplate
from wiki_evaluation import WikiEvaluation
from wiki_correctness import WikiCorrectness

# --- CẤU HÌNH CHUNG ---
BASE_PATH = "data_storage"
RAW_SOURCE_DIR = os.path.join(BASE_PATH, "raw") 
LLM_CONFIG_FILE = "LLM.txt"
TEMPLATE_FILE = "crop_standard.json"
TEST_DATA_DIR = "test_data"

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def load_llm_config():
    config = {}
    try:
        with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip()
        return config
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {LLM_CONFIG_FILE}")
        return None

def evaluate_wiki(session_id):
    """
    Quy trình đánh giá toàn diện: Citation (NLI) & Correctness (LLM + NLI)
    """
    print(f"\nBẮT ĐẦU ĐÁNH GIÁ TỔNG THỂ: {session_id}")
    
    # 1. KHỞI TẠO ĐƯỜNG DẪN
    session_dir = f"output/{session_id}"
    array_file = os.path.join(session_dir, f"{session_id}_array.json")
    bib_file = os.path.join(session_dir, f"{session_id}_bibliography.json")
    ai_txt_file = os.path.join(session_dir, f"{session_id}.txt")
    base_txt_file = "base.txt" # Bài mẫu của người viết

    # 2. KHỞI ĐỘNG LLM (Dùng cho WikiCorrectness để tách Atomic Claims)
    cfg = load_llm_config()
    llm = LLMManager(
        provider=cfg.get("Provider"),
        base_url=cfg.get("Base URL"),
        api_key=cfg.get("API KEY"),
        model_name=cfg.get("Model Name")
    )

    # ---------------------------------------------------------
    # PHẦN 1: ĐÁNH GIÁ TRÍCH DẪN (CITATION RECALL & PRECISION)
    # ---------------------------------------------------------
    print("\n--- 1. Đang đánh giá Citation (Chất lượng trích dẫn) ---")
    try:
        with open(array_file, 'r', encoding='utf-8') as f:
            article_array = json.load(f)
        with open(bib_file, 'r', encoding='utf-8') as f:
            array_bibliography = json.load(f)

        evaluator = WikiEvaluation(
            article_array=article_array, 
            source_dir_path=RAW_SOURCE_DIR, 
            session_id=session_id,
            array_bibliography=array_bibliography
        )
        evaluator.preprocess()

        cit_recall = evaluator.calculate_avg_citation_recall()
        cit_precision = evaluator.calculate_avg_citation_precision()
    except Exception as e:
        print(f"❌ Lỗi trong quá trình Citation Evaluation: {e}")
        cit_recall, cit_precision = 0, 0

    # ---------------------------------------------------------
    # PHẦN 2: ĐÁNH GIÁ NỘI DUNG (CORRECTNESS / CLAIM RECALL)
    # ---------------------------------------------------------
    print("\n--- 2. Đang đánh giá Correctness (Độ chính xác nội dung) ---")
    try:
        if os.path.exists(ai_txt_file) and os.path.exists(base_txt_file):
            with open(ai_txt_file, 'r', encoding='utf-8') as f:
                ai_text = f.read()
            with open(base_txt_file, 'r', encoding='utf-8') as f:
                base_text = f.read()

            correctness_eval = WikiCorrectness(
                article_to_check=ai_text,
                reference_article=base_text,
                llm=llm,
                output_path=session_dir # Lưu debug log vào folder session
            )
            
            # Tách ý và tính điểm
            correctness_eval.extract_atomic_claims()
            claim_recall = correctness_eval.calculate_correctness()
        else:
            print("⚠️ Không tìm thấy file bài AI hoặc bài mẫu (base.txt). Bỏ qua Correctness.")
            claim_recall = 0
    except Exception as e:
        print(f"❌ Lỗi trong quá trình Correctness Evaluation: {e}")
        claim_recall = 0

    # ---------------------------------------------------------
    # TỔNG HỢP VÀ LƯU KẾT QUẢ
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ CUỐI CÙNG CHO SESSION: {session_id}")
    print("-" * 50)
    print(f"1. Citation Recall (Trung thực):    {cit_recall:.2f}%")
    print(f"2. Citation Precision (Súc tích):   {cit_precision:.2f}%")
    print(f"3. Claim Recall (Độ đầy đủ ý):      {claim_recall*100:.2f}%")
    print("="*50)

    # Lưu vào file JSON tổng hợp
    final_result_path = os.path.join(session_dir, f"{session_id}_final_metrics.json")
    with open(final_result_path, 'w', encoding='utf-8') as f:
        json.dump({
            "citation_recall": cit_recall,
            "citation_precision": cit_precision,
            "claim_recall": claim_recall * 100
        }, f, indent=4)
    print(f"💾 Đã lưu tất cả chỉ số vào: {final_result_path}")

# (Giữ nguyên các hàm write(), load_template_from_json() của bạn ở dưới...)

if __name__ == "__main__":
    # Test thử với session bạn yêu cầu
    evaluate_wiki('mshveifttbdx')