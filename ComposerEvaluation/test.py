import torch
from sentence_transformers import CrossEncoder

class CitationEvaluator:
    def __init__(self):
        print("--- Đang tải model NLI (DeBERTa-v3-base)... ---")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrossEncoder('cross-encoder/nli-deberta-v3-base', device=device)
        print(f"✅ Đã tải xong trên: {device}")

    def _check_entailment(self, premise, hypothesis):
        if not premise.strip(): 
            return False
        scores = self.model.predict([(premise, hypothesis)])
        return scores[0].argmax() == 1  # 1 là Entailment

    # --- CÔNG THỨC 1: CITATION RECALL ---
    def calculate_citation_recall(self, statement, cited_chunks):
        if not cited_chunks: return 0
        combined_premise = " ".join(cited_chunks)
        is_entailed = self._check_entailment(combined_premise, statement)
        return 1 if is_entailed else 0

    # --- CÔNG THỨC 2: CITATION PRECISION ---
    def calculate_citation_precision(self, statement, cited_chunks):
        # [FIX LỖI CRASH]: Kiểm tra Recall trước
        # Nếu Recall = 0 -> Precision = 0. Trả về đúng format (score, list)
        if self.calculate_citation_recall(statement, cited_chunks) == 0:
            return 0.0, ["Recall = 0 (Toàn bộ nguồn không đủ chứng minh)"]

        valid_citations_count = 0
        total_citations = len(cited_chunks)
        details = []

        for i, target_citation in enumerate(cited_chunks):
            # Điều kiện (a): Bản thân nó KHÔNG entail statement (Nó yếu)
            condition_a = not self._check_entailment(target_citation, statement)
            
            # Điều kiện (b): Các thằng CÒN LẠI (sau khi bỏ nó ra) VẪN entail statement (Nó vô dụng)
            remaining_chunks = [c for j, c in enumerate(cited_chunks) if j != i]
            combined_remaining = " ".join(remaining_chunks)
            
            if not remaining_chunks:
                condition_b = False # Nếu bỏ nó ra mà hết sạch nguồn thì chắc chắn còn lại = False
            else:
                condition_b = self._check_entailment(combined_remaining, statement)
            
            # KẾT LUẬN: Nó thừa khi (Nó yếu) VÀ (Nhóm còn lại vẫn mạnh)
            if condition_a and condition_b:
                is_precise = 0
                details.append(f"Source [{i+1}]: THỪA (Irrelevant - Bỏ đi vẫn chứng minh được)")
            else:
                is_precise = 1
                valid_citations_count += 1
                details.append(f"Source [{i+1}]: CẦN THIẾT (Precise - Đóng góp thông tin quan trọng)")
                
        precision_score = valid_citations_count / total_citations
        return precision_score, details

# ==========================================
# KHU VỰC CHẠY THỬ (Đã điều chỉnh ví dụ chuẩn logic)
# ==========================================
if __name__ == "__main__":
    evaluator = CitationEvaluator()
    
    print("\n" + "="*50)
    print("KIỂM TRA CÔNG THỨC (ĐÃ FIX)")
    print("="*50)

    # --- CASE 1: CẢ 2 NGUỒN ĐỀU CẦN THIẾT (Precision = 1.0) ---
    # Để đạt 1.0, mỗi nguồn phải chứa 1 nửa thông tin mà nguồn kia không có
    statement_1 = "Ông Hồ Quang Cua lai tạo lúa ST25 tại Sóc Trăng."
    chunks_1 = [
        "Ông Hồ Quang Cua là cha đẻ của giống lúa ST25.",         # (Thiếu Sóc Trăng -> Bỏ ra thì thiếu ý -> Cần thiết)
        "Quá trình lai tạo ST25 diễn ra tại tỉnh Sóc Trăng."      # (Thiếu ông Cua -> Bỏ ra thì thiếu ý -> Cần thiết)
    ]
    
    print("\n🔹 CASE 1: Nguồn bổ sung cho nhau (High Precision)")
    rec_1 = evaluator.calculate_citation_recall(statement_1, chunks_1)
    prec_1, det_1 = evaluator.calculate_citation_precision(statement_1, chunks_1)
    print(f"Statement: {statement_1}")
    print(f"-> Recall:    {rec_1} (Mong đợi: 1)")
    print(f"-> Precision: {prec_1} (Mong đợi: 1.0)")
    print(f"-> Chi tiết:  {det_1}")

    # --- CASE 2: CÓ NGUỒN THỪA (Precision = 0.5) ---
    # Nguồn 1 đã đủ chứng minh, nên Nguồn 2 bị coi là thừa
    statement_2 = "Gạo ST25 đạt giải nhất năm 2019."
    chunks_2 = [
        "Năm 2019, gạo ST25 xuất sắc giành giải gạo ngon nhất thế giới.", # (Đủ ý -> Cần thiết)
        "Gạo ST25 rất thơm và dẻo."                                      # (Không liên quan năm/giải -> Thừa)
    ]
    
    print("\n🔹 CASE 2: Có nguồn thừa (Irrelevant)")
    rec_2 = evaluator.calculate_citation_recall(statement_2, chunks_2)
    prec_2, det_2 = evaluator.calculate_citation_precision(statement_2, chunks_2)
    print(f"Statement: {statement_2}")
    print(f"-> Recall:    {rec_2} (Mong đợi: 1)")
    print(f"-> Precision: {prec_2} (Mong đợi: 0.5)")
    print(f"-> Chi tiết:  {det_2}")

    # --- CASE 3: BỊA ĐẶT (Recall = 0 -> Precision = 0) ---
    # Đã fix lỗi crash ở đây
    statement_3 = "Gạo ST25 có nguồn gốc từ Thái Lan."
    chunks_3 = [
        "Gạo ST25 là niềm tự hào của Việt Nam.",
        "Sóc Trăng là nơi ra đời của ST25."
    ]
    
    print("\n🔹 CASE 3: Bịa đặt (Hallucination)")
    rec_3 = evaluator.calculate_citation_recall(statement_3, chunks_3)
    prec_3, det_3 = evaluator.calculate_citation_precision(statement_3, chunks_3)
    print(f"Statement: {statement_3}")
    print(f"-> Recall:    {rec_3} (Mong đợi: 0)")
    print(f"-> Precision: {prec_3} (Mong đợi: 0.0)")
    print(f"-> Chi tiết:  {det_3}")