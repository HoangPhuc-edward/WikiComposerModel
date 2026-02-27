import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate

# [UPDATED] Import theo chuẩn mới của Ragas để tránh DeprecationWarning
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

# Import wrapper của LangChain cho Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- CẤU HÌNH ---
# Điền API Key của bạn vào đây
API_KEY = "AIzaSyBCaA_s2lkjBgxFLQkzItRqmOyRzCq89Cg" # <--- THAY API KEY CỦA BẠN VÀO ĐÂY
os.environ["GOOGLE_API_KEY"] = API_KEY

def setup_gemini():
    """Khởi tạo LLM và Embeddings"""
    # 1. LLM Giám khảo
    # Lưu ý: Thêm prefix 'models/' để định danh chính xác hơn
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        max_output_tokens=1024
    )

    # 2. Embedding để tính toán vector
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    return llm, embeddings

def test_connection(llm):
    """Hàm kiểm tra kết nối tới Gemini trước khi chạy Ragas"""
    print("\n--- BƯỚC 1: TEST KẾT NỐI GEMINI ---")
    try:
        response = llm.invoke("Xin chào, bạn có hoạt động không?")
        print(f"✅ Kết nối thành công! Phản hồi: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Kết nối thất bại. Lỗi chi tiết:\n{e}")
        return False

def run_evaluation():
    # 1. Khởi tạo
    gemini_llm, gemini_embeddings = setup_gemini()

    # 2. Test kết nối (QUAN TRỌNG)
    if not test_connection(gemini_llm):
        return

    # 3. Chuẩn bị dữ liệu mẫu
    print("\n--- BƯỚC 2: CHUẨN BỊ DỮ LIỆU ---")
    data_samples = {
        'question': [
            'Gạo ST25 do ai lai tạo?',
            'Cách bón phân cho lúa ST25?'
        ],
        'answer': [
            'Gạo ST25 do kỹ sư Hồ Quang Cua lai tạo.',
            'Cần bón nhiều phân đạm.' # Cố tình sai để check faithfulness
        ],
        'contexts': [
            ['Gạo ST25 là giống lúa thơm Sóc Trăng. Cha đẻ là kỹ sư Hồ Quang Cua.'],
            ['Lúa ST25 cần bón cân đối. Không nên bón thừa đạm.']
        ],
        'ground_truth': [
            'Kỹ sư Hồ Quang Cua.',
            'Bón phân cân đối, tránh thừa đạm.'
        ]
    }
    dataset = Dataset.from_dict(data_samples)

    # 4. Chạy đánh giá
    print("\n--- BƯỚC 3: ĐANG CHẤM ĐIỂM BẰNG RAGAS (GEMINI) ---")
    
    # [UPDATED] Khởi tạo metrics dạng Class
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision()
    ]

    try:
        results = evaluate(
            dataset,
            metrics=metrics,
            llm=gemini_llm,       # Gemini làm giám khảo
            embeddings=gemini_embeddings # Gemini tính vector
        )
        
        print("\n✅ ĐÁNH GIÁ HOÀN TẤT!")
        print("\n--- KẾT QUẢ TỔNG HỢP ---")
        print(results)
        
        print("\n--- CHI TIẾT TỪNG CÂU ---")
        df = results.to_pandas()
        # In ra các cột quan trọng
        print(df[['question', 'faithfulness', 'answer_relevancy', 'context_precision']])

        # Lưu file excel nếu cần
        # df.to_excel("ragas_results.xlsx", index=False)

    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình đánh giá Ragas: {e}")

if __name__ == "__main__":
    run_evaluation()