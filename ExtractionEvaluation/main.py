from extractor import Extractor
import json


with open('output1.json', 'w', encoding='utf-8') as f:
    # web_link = "https://vnexpress.net/cam-17-5-cua-tap-doan-nongfu-spring-ra-mat-tai-viet-nam-4993820.html"
    # content = Extractor().extract_website(web_link)
    # json.dump({"source": web_link, "content": content}, f, ensure_ascii=False, indent=2)

    doc_link = "datasets/text_files/file/Chôm chôm Dona.docx"
    content = Extractor().extract_text_file(doc_link)
    json.dump({"source": doc_link, "content": content}, f, ensure_ascii=False, indent=2)

    pdf_link = "datasets/text_files/file/Bưởi da xanh.pdf"
    content = Extractor().extract_text_file(pdf_link)
    json.dump({"source": pdf_link, "content": content}, f, ensure_ascii=False, indent=2)

    # youtube_link = "https://www.youtube.com/watch?v=vmwANBlSgOw"
    # content = Extractor().extract_youtube(youtube_link)
    # json.dump({"source": youtube_link, "content": content}, f, ensure_ascii=False, indent=2)