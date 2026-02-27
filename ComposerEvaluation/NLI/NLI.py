from sentence_transformers import CrossEncoder
import torch

class CitationValidator:
    def __init__(self):
        print("--- Đang tải model NLI (DeBERTa-v3-base)... ---")
        # Tự động tải model về cache nếu chưa có (khoảng 800MB)
        # Sử dụng GPU nếu có, không thì dùng CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrossEncoder('cross-encoder/nli-deberta-v3-base', device=device)
        print(f"✅ Đã tải xong model trên thiết bị: {device}")

        # Mapping nhãn của model này
        self.label_mapping = ['Contradiction', 'Entailment', 'Neutral']

    def check_citation(self, source_text, generated_claim):
        """
        Kiểm tra xem source_text có hỗ trợ generated_claim hay không.
        Trả về:
            - is_supported (bool): True nếu Entailment
            - score (float): Điểm tin cậy (0.0 đến 1.0)
            - label (str): Nhãn dự đoán
        """
        # CrossEncoder nhận input là list các cặp [(Câu A, Câu B)]
        scores = self.model.predict([(source_text, generated_claim)])
        
        # scores là logits, ta lấy argmax để tìm nhãn có điểm cao nhất
        pred_label_index = scores[0].argmax()
        pred_label = self.label_mapping[pred_label_index]
        
        # Tính xác suất (Softmax) để lấy độ tin cậy (Option phụ để hiển thị cho đẹp)
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=1).numpy()[0]
        confidence = probs[pred_label_index]

        # QUY TẮC ĐÁNH GIÁ:
        # Chỉ trả về True nếu nhãn là 'Entailment' (Index = 1)
        is_supported = (pred_label_index == 1)

        return {
            "is_supported": is_supported,
            "confidence": float(confidence),
            "label": pred_label
        }

# --- PHẦN TEST THỬ (Chạy trực tiếp file này để xem) ---
if __name__ == "__main__":
    validator = CitationValidator()

    print("\n--- TEST CASE 1: ĐÚNG ---")
    nguon = "Gạo ST25 được lai tạo bởi kỹ sư Hồ Quang Cua và nhóm cộng sự tại Sóc Trăng."
    cau_ai_viet = "Ông Hồ Quang Cua là cha đẻ của giống lúa ST25."
    result = validator.check_citation(nguon, cau_ai_viet)
    print(f"Kết quả: {result['label']} ({result['confidence']:.2f}) -> Supported: {result['is_supported']}")

    print("\n--- TEST CASE 2: BỊA ĐẶT (Hallucination) ---")
    nguon = "Gạo ST25 được lai tạo bởi kỹ sư Hồ Quang Cua."
    cau_ai_viet = "Gạo ST25 có nguồn gốc xuất xứ từ Thái Lan."
    result = validator.check_citation(nguon, cau_ai_viet)
    print(f"Kết quả: {result['label']} ({result['confidence']:.2f}) -> Supported: {result['is_supported']}")

    print("\n--- TEST CASE 3: KHÔNG LIÊN QUAN (Neutral) ---")
    nguon = "Gạo ST25 ngon nhất thế giới năm 2019."
    cau_ai_viet = "Gạo ST25 có giá bán rất cao." 
    # (Nguồn chỉ nói ngon, không nói giá -> Suy diễn -> Neutral)
    result = validator.check_citation(nguon, cau_ai_viet)
    print(f"Kết quả: {result['label']} ({result['confidence']:.2f}) -> Supported: {result['is_supported']}")