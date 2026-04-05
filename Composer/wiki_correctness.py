import os
import time
import torch
import re
from sentence_transformers import CrossEncoder

class WikiCorrectness:
    def __init__(self, article_to_check, reference_article, llm, output_path, nli_model=None):
        """
        :param article_to_check: Bài viết A (AI viết) - Đóng vai trò Premise
        :param reference_article: Bài mẫu B (Người viết/Chuẩn) - Đóng vai trò nguồn để tách Claims
        :param llm: Instance của LLMManager để gọi GPT/Groq
        :param output_path: Đường dẫn thư mục để lưu debug log
        """
        self.article_a = article_to_check
        self.article_b = reference_article
        self.llm = llm
        self.output_path = output_path
        
        # Đảm bảo thư mục output tồn tại
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Khởi tạo mô hình NLI (Giống WikiEvaluation)
        if nli_model:
            self.nli_model = nli_model
        else:
            self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

        self.atomic_claims = []

    def extract_atomic_claims(self):
        print(f"--- Đang tách Atomic Claims từ bài mẫu B ({len(self.article_b)} ký tự)... ---")
        
        # 1. Chia nhỏ article_b thành các chunk khoảng 4000 ký tự
        chunks = []
        current_chunk = ""
        for line in self.article_b.split('\n'):
            if len(current_chunk) + len(line) + 1 > 4000:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line
        if current_chunk:
            chunks.append(current_chunk)

        self.atomic_claims = []
        
        # 2. Duyệt qua từng chunk để gửi cho LLM
        for i, text_segment in enumerate(chunks):
            print(f"   -> Đang xử lý đoạn {i+1}/{len(chunks)}...")
            
            prompt = f"""
            Nhiệm vụ: Hãy tách đoạn văn bản sau đây thành danh sách các 'Atomic Claims' (câu khẳng định đơn cực ngắn).
            Yêu cầu:
            1. Mỗi câu chỉ chứa duy nhất MỘT thông tin/ý khẳng định.
            2. Không sử dụng các từ nối phức tạp (và, nhưng, vì, nên...).
            3. Giữ nguyên các danh từ riêng, con số, thuật ngữ chuyên môn.
            4. Trả về kết quả dưới dạng danh sách gạch đầu dòng.

            VĂN BẢN CẦN TÁCH:
            {text_segment}
            
            DANH SÁCH ATOMIC CLAIMS:
            """

            # Logic retry khi gặp lỗi API hoặc Rate Limit
            idx = 0
            response = ""
            while True:
                response = self.llm.send_prompt(prompt[:12000], options={"temperature": 0.1})

                if "error" in response.lower():
                    if idx >= 4 or "429" in response.lower():  
                        print("Không thể viết LLM: quá token")
                        raise Exception("Không thể viết LLM: quá token")

                    else:
                        print("Đang chờ 60 giây do quá tải LLM...")
                        time.sleep(60)
                        idx +=1

                else:
                    break
                 

            # 3. Xử lý response của từng đoạn và gộp vào danh sách chung
            if response:
                raw_claims = response.strip().split('\n')
                for line in raw_claims:
                    clean_claim = re.sub(r'^[\s\-\d\.]+', '', line).strip()
                    if clean_claim:
                        self.atomic_claims.append(clean_claim)

        # 4. Lưu toàn bộ kết quả vào debug log
        debug_file = os.path.join(self.output_path, "correctness_debug.txt")
        try:
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"=== ATOMIC CLAIMS EXTRACTED FROM ARTICLE B ===\n")
                f.write(f"Total Claims: {len(self.atomic_claims)}\n\n")
                for j, claim in enumerate(self.atomic_claims):
                    f.write(f"{j+1}. {claim}\n")
                f.write("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"❌ Lỗi ghi file debug: {e}")
        
        print(f"✅ Hoàn thành! Đã tách được tổng cộng {len(self.atomic_claims)} claims. Xem tại: {debug_file}")
        return self.atomic_claims

    def calculate_correctness(self):
        """
        Thuật toán chấm điểm Correctness (Claim Recall)
        Premise: Bài A (AI viết)
        Hypothesis: Từng Atomic Claim của bài B
        """
        if not self.atomic_claims:
            self.extract_atomic_claims()

        total_claims = len(self.atomic_claims)
        entailed_count = 0
        debug_results = []

        print(f"--- Đang kiểm tra NLI cho {total_claims} claims... ---")

        for i, claim in enumerate(self.atomic_claims):
            # Premise = Bài viết của AI (A)
            # Hypothesis = Claim i của bài mẫu (B)
            scores = self.nli_model.predict([(self.article_a, claim)])
            label_index = scores[0].argmax()
            
            # Label mapping: 0: Contradiction, 1: Entailment, 2: Neutral
            is_entailed = (label_index == 1)
            
            status = "PASS" if is_entailed else "FAIL"
            if is_entailed:
                entailed_count += 1
            
            # Lưu log chi tiết cho từng claim
            debug_results.append(f"Claim {i+1}: {claim}\nResult: {status} (Label: {label_index})\n")

        # Tính toán Claim Recall
        score = (entailed_count / total_claims) if total_claims > 0 else 0
        
        # Ghi log kết quả chi tiết
        debug_file = os.path.join(self.output_path, "correctness_debug.txt")
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("\n=== NLI VERIFICATION RESULTS (A covers B?) ===\n")
            f.writelines(debug_results)
            f.write(f"\nFINAL CLAIM RECALL SCORE: {score:.4f} ({entailed_count}/{total_claims})\n")

        print(f"📊 Correctness Score: {score*100:.2f}%")
        return score

    def debug(self, path):
        """Lưu trữ danh sách Atomic Claims và các bài viết đối chiếu"""
        debug_data = {
            "ai_article_premise": self.article_a,
            "reference_article_source": self.article_b,
            "extracted_atomic_claims": self.atomic_claims
        }
        import json
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, ensure_ascii=False, indent=4)
        print(f"📄 [Debug] Đã lưu dữ liệu WikiCorrectness tại: {path}")