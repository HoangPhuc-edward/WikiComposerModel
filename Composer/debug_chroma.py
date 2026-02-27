import json
import os

def analyze_wiki_quantities(file_path):
    """
    Đọc file JSONL và thống kê số lượng ký tự, token của fullContent.
    """
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file {file_path}")
        return

    results = []
    grand_total_chars = 0
    grand_total_tokens = 0

    print(f"📊 ĐANG PHÂN TÍCH FILE: {file_path}")
    print("-" * 70)
    print(f"{'STT':<5} | {'Ký tự':<10} | {'Tokens*':<10} | {'URL'}")
    print("-" * 70)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Mỗi dòng là một object JSON
                data = json.loads(line)
                content = data.get('fullContent', "")
                url = data.get('url', 'N/A')

                # 1. Đếm số ký tự
                char_count = len(content)
                
                # 2. Ước tính token (Định lượng bằng số từ tách theo khoảng trắng)
                # Đây là cách tính gần đúng phổ biến cho tiếng Việt/Anh
                token_count = len(content.split())

                results.append({
                    "stt": i + 1,
                    "char_count": char_count,
                    "token_count": token_count,
                    "url": url
                })

                grand_total_chars += char_count
                grand_total_tokens += token_count

                print(f"{i+1:<5} | {char_count:<10} | {token_count:<10} | {url}")

    except Exception as e:
        print(f"❌ Có lỗi xảy ra trong quá trình đọc: {e}")
        return

    # --- TỔNG KẾT ---
    print("-" * 70)
    print(f"📌 TỔNG CỘNG ({len(results)} bài viết):")
    print(f"   - Tổng số ký tự: {grand_total_chars:,}")
    print(f"   - Tổng số tokens (ước tính): {grand_total_tokens:,}")
    print(f"   - Trung bình: ~{int(grand_total_tokens/len(results)) if results else 0} tokens/bài")
    print("-" * 70)
    print("(*) Ghi chú: Token được định lượng bằng số lượng từ (split by whitespace).")

if __name__ == "__main__":
    # Thay đổi tên file cho đúng với thực tế của bạn
    # Lưu ý: 'jsonl' là định dạng mỗi dòng một JSON object
    FILE_NAME = "uncleaned_data.jsonl" 
    analyze_wiki_quantities(FILE_NAME)