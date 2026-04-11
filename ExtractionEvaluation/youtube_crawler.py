
from docx import Document
import os
import csv

list_files = os.listdir("datasets/text_files/file")

csv_path = 'datasets/text_files/train.csv'
file_exists = os.path.isfile(csv_path)
with open(csv_path, mode='a', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['file_name', 'content'])

    idx = 0
    for file in list_files:
        idx += 1
        if file.endswith(".docx"):
            doc = Document(os.path.join("datasets/text_files/file", file))
            paragraphs = [p.text for p in doc.paragraphs]
            text_content = "\n".join(paragraphs)
            writer.writerow([file.replace(".docx", ""), text_content])

print(f"Đã xử lý {idx} file và lưu vào {csv_path}")