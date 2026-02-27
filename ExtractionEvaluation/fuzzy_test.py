from thefuzz import fuzz
import unicodedata
import re

def clean_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

ground_truth = "loại"
extracted    = "lại"

s1 = clean_text(ground_truth)
s2 = clean_text(extracted)

print(f"Độ dài chuỗi 1: {len(s1)}")
print(f"Độ dài chuỗi 2: {len(s2)}")
print("-" * 30)

hard_result = (s1 == s2)
print(f"[So sánh cứng] (s1 == s2): {hard_result}")
# -> Kết quả: False

fuzzy_score = fuzz.ratio(s1, s2)
print(f"[So sánh Fuzzy] Điểm số: {fuzzy_score}/100")


threshold = 85
if fuzzy_score >= threshold:
    print(f"-> KẾT LUẬN: ĐÚNG (Vì {fuzzy_score}% >= {threshold}%)")
else:
    print(f"-> KẾT LUẬN: SAI")