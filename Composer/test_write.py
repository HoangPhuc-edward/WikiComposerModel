
import json

from llm_engine import LLMManager
from wiki_composer import WikiComposer
from preprocessor import Preprocessor
from template_manager import ContentTemplate
import os

my_template = ContentTemplate(
    name="Template bài viết về gạo ST25",
    description="Template này được thiết kế để tạo ra một bài viết chi tiết về giống lúa ST25, bao gồm các phần như giới thiệu, đặc điểm nổi bật, quy trình trồng trọt, và lợi ích của giống lúa này.",
    system_instruction="Bạn là một chuyên gia về nông nghiệp, hãy viết một bài wiki vừa cô đọng vừa đơn giản",
    structure=[
      {
        "title": "Giới thiệu chung",
        "description": "Giới thiệu chung về giống lúa ST25"
      },
      {
        "title": "Nguồn gốc",
        "description": "Nguồn gốc tên gọi và người tạo ra giống lúa ST25."
      },
      {
        "title": "Tác giả Hồ Quang Cua",
        "description": "Thông tin về ông Hồ Quang Cua"
      }
    ]
)



def load_llm_config():
    
    config = {}
    try:
        with open("LLM.txt", "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip()
        return config
    except Exception as e:
        print(f"Lỗi nạp config: {e}"); return None


cfg = load_llm_config()
my_llm = LLMManager(provider=cfg.get("Provider"), base_url=cfg.get("Base URL"), api_key=cfg.get("API KEY"), model_name=cfg.get("Model Name"))

my_composer = WikiComposer(session_id="1223_24022025", name="Lúa ST25", template=my_template, llm=my_llm)

# Preprocess the input
my_preproccessor = Preprocessor(session_id="1223_24022025")
try:
    my_preproccessor.execute("lua_st25/Lúa ST25.docx", input_type="docx")
    my_preproccessor.execute("https://www.youtube.com/watch?v=YgZNTqDDr4U", input_type="youtube")
    print("Preprocessing completed successfully.")
except Exception as e:
    print(f"Error during preprocessing: {e}")

# Write the wiki
try:
    output = my_composer.wiki_compose()
    session_id = my_composer.session_id
    
    path = f"output/{session_id}"
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{session_id}.txt"), "w", encoding="utf-8") as f: f.write(output["full_content"])
    with open(os.path.join(path, f"{session_id}_array.json"), "w", encoding="utf-8") as f: json.dump(output["array_content"], f, ensure_ascii=False, indent=4)
    with open(os.path.join(path, f"{session_id}_bibliography.json"), "w", encoding="utf-8") as f: json.dump(output["array_bibliography"], f, ensure_ascii=False, indent=4)
        
    print(f"Đã lưu dữ liệu bài {session_id} vào {path}")
except Exception as e:
    print(f"Error during wiki composition: {e}")