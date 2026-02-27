import os
import json
import random
import re
from typing import List, Dict, Set, Optional, Any, Tuple
import chromadb
from sentence_transformers import SentenceTransformer

# Import các class phụ thuộc (giữ nguyên cấu trúc hiện tại của bạn)
from template_manager import ContentTemplate
from llm_engine import LLMManager
from sentence_transformers import CrossEncoder


class WikiComposer:
    def __init__(self, 
                 session_id: str, 
                 name: str, 
                 template: ContentTemplate, 
                 llm: LLMManager, 
                 base_dir: str = "data_storage", 
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        
        self.session_id = session_id
        self.name = name
        self.template = template
        self.llm = llm
        self.base_dir = base_dir

        self.vector_path = os.path.join(base_dir, "vector_db")
        self.source_file = os.path.join(base_dir, "source.json")
        
        self.chroma_client = chromadb.PersistentClient(path=self.vector_path)
        self.collection = self.chroma_client.get_or_create_collection(name="wiki_docs")
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.full_content: str = ""
        self.array_content: List[Dict] = []
        self.bibliography: str = ""
        self.array_bibliography: List[Dict] = []

        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.context: str = self._fetch_random_context()[:600]

    def _fetch_random_context(self) -> str:
        try:
            results = self.collection.get(where={"doc_name": self.session_id}, limit=20)
            docs = results.get("documents", [])
            if not docs: return ""
            k = min(10, len(docs))
            random_docs = random.sample(docs, k)
            return "\n...\n".join(random_docs)
        except Exception: return ""

    def _get_relevant_chunks(self, title: str) -> List[Dict]:
        """
        Quy trình: Retrieve (Lấy 20) -> Rerank -> Top-K (Lấy 5)
        """
        # 1. Embed câu query
        query_vector = self.embedding_model.encode(title).tolist()
        
        # 2. Giai đoạn 1: RETRIEVAL (Lấy số lượng lớn ứng viên, VD: 20)
        # Bỏ vòng lặp while cũ, lấy thẳng top 20 vector gần nhất
        initial_k = 20 
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=initial_k, 
            where={"doc_name": self.session_id}
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        doc_list = results["documents"][0]
        meta_list = results["metadatas"][0]
        
        # 3. Giai đoạn 2: RERANKING
        # Tạo các cặp [Query, Document] để Cross-Encoder chấm điểm
        rerank_pairs = [[title, doc_text] for doc_text in doc_list]
        
        # Model chấm điểm (Trả về logits, càng cao càng liên quan)
        scores = self.reranker.predict(rerank_pairs)
        
        # 4. Giai đoạn 3: SẮP XẾP & CHỌN LỌC
        # Kết hợp: (Điểm, Nội dung, Metadata)
        ranked_results = []
        for score, doc, meta in zip(scores, doc_list, meta_list):
            ranked_results.append({
                "score": score,
                "content": doc,
                "metadata": meta
            })
            
        # Sắp xếp giảm dần theo điểm (Score cao nhất lên đầu)
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Lấy Top 5 kết quả tốt nhất
        final_top_k = 5
        top_results = ranked_results[:final_top_k]
        
        # 5. Format lại dữ liệu trả về theo đúng cấu trúc cũ
        relevant_data = []
        for item in top_results:
            # Có thể thêm ngưỡng lọc nếu muốn (VD: score > 0)
            # if item["score"] < -2: continue 
            
            relevant_data.append({
                "content": item["content"],
                "metadata": item["metadata"]
            })
            
        return relevant_data

    def _query_expansion(self, title: str, description: str) -> str:
        """
        Sử dụng LLM để biến đổi tiêu đề ngắn thành một câu truy vấn đầy đủ ngữ nghĩa,
        bao gồm các từ khóa tiềm năng để tăng độ chính xác khi tìm kiếm Vector.
        """
        prompt = f"""
        ### VAI TRÒ:
        Bạn là một chuyên gia tìm kiếm thông tin và tối ưu hóa truy vấn (SEO).
        
        ### THÔNG TIN BÀI VIẾT:
        - Tên bài: {self.name}
        - Bối cảnh chung: {self.context}
        
        ### MỤC CẦN VIẾT:
        - Tiêu đề mục: "{title}"
        - Mô tả yêu cầu: {description}
        
        ### NHIỆM VỤ:
        Hãy viết lại một "câu truy vấn tìm kiếm" (search query) để tìm thông tin cho mục này trong cơ sở dữ liệu.
        
        ### YÊU CẦU:
        1. Câu truy vấn nên gợi ý sâu hơn vào mô tả yêu cầu, có thể bao gồm các từ khóa liên quan, khái niệm chuyên ngành, hoặc cách diễn đạt khác của tiêu đề.
        2. Bổ sung các từ khóa, khái niệm chuyên ngành liên quan có thể xuất hiện trong tài liệu nguồn.
        3. Viết dưới dạng một đoạn văn ngắn hoặc các câu nối tiếp nhau, khoảng 30-50 từ.
        4. CHỈ TRẢ VỀ NỘI DUNG CÂU TRUY VẤN, KHÔNG GIẢI THÍCH.
        5. KHÔNG GHI NGUYÊN BỐI CẢNH VÀO CÂU TRUY VẤN, CHỈ TẬP TRUNG VÀO VIỆC MỞ RỘNG TIÊU ĐỀ VÀ MÔ TẢ YÊU CẦU THÀNH CÂU TRUY VẤN TÌM KIẾM.
        """
        
        # Gọi LLM (có thể để temperature cao hơn một chút để AI "sáng tạo" từ khóa)
        try:
            expanded_query = self.llm.send_prompt(prompt, options={"temperature": 0.3})
            # Làm sạch cơ bản (bỏ dấu ngoặc kép nếu có)
            return expanded_query.strip().strip('"').strip("'")
        except Exception as e:
            print(f"⚠️ Lỗi Query Expansion: {e}")
            # Fallback về tiêu đề gốc nếu lỗi
            return f"{self.name} {title} {description}"

    def write_section(self, title: str, description: str) -> Tuple[str, List[Dict]]:
        """Viết bài và trả về danh sách metadata nguồn thô"""
        
        # [BƯỚC 1] Query Expansion: Tạo câu truy vấn giàu ngữ nghĩa
        search_query = self._query_expansion(title, description)
        print(f"   -> Expanded Query cho '{title}': {search_query[:100]}...") # Log để debug

        # [BƯỚC 2] Tìm kiếm chunk bằng query đã mở rộng
        chunks = self._get_relevant_chunks(search_query)
        
        if not chunks:
            return "Chưa có thông tin cập nhật cho mục này.", []

        context_str = ""
        raw_sources_meta = []
        for item in chunks:
            context_str += f" {item['content']}"
            raw_sources_meta.append(item['metadata'])

        # [BƯỚC 3] Viết nội dung (Logic cũ)
        prompt = f"""
            ### VAI TRÒ:
            {self.template.system_instruction}

            ### BỐI CẢNH TOÀN BÀI:
            {self.context}

            ### DỮ LIỆU THAM KHẢO:
            {context_str}

            ### NHIỆM VỤ:
            Viết nội dung cho mục: "{title}".
            Hướng dẫn chi tiết: {description}

            ### YÊU CẦU:
            1. Tổng hợp thông tin từ dữ liệu tham khảo thành đoạn văn xuôi mượt mà.
            2. Tuyệt đối KHÔNG sử dụng ký tự Markdown (*, #, **, __).
            3. KHÔNG sử dụng dấu xuống dòng (\\n) bên trong đoạn văn.
            4. Nếu dữ liệu không đủ, chỉ viết những gì có chắc chắn.
            """
        
        response = self.llm.send_prompt(prompt, options={"temperature": 0.1})
        
        # Làm sạch văn bản triệt để
        clean_text = response.strip()
        clean_text = re.sub(r'[\*\#\_]', '', clean_text) # Xóa *, #, _
        clean_text = re.sub(r'\s+', ' ', clean_text)     # Biến mọi loại khoảng trắng/xuống dòng thành 1 khoảng trắng
        
        return clean_text, raw_sources_meta
  

    def _dfs_traverse(self, node: Dict) -> Dict:
        """Duyệt DFS: Đối chiếu locator object và gán STT nguồn"""
        title = node.get("title", "Mục không tên")
        result_node = {
            "title": title,
            "type": "section",
            "content": "",
            "source": [],
            "children": []
        }

        if "subsections" in node and isinstance(node["subsections"], list) and len(node["subsections"]) > 0:
            full_text_children = ""
            for child in node["subsections"]:
                child_result = self._dfs_traverse(child)
                result_node["children"].append(child_result)
                if child_result["content"]:
                    full_text_children += f"\n{child_result['title']}\n{child_result['content']}\n"
            result_node["content"] = full_text_children
        
        elif "description" in node:
            content, raw_sources_meta = self.write_section(title, node["description"])
            
            # Xử lý mapping nguồn (Bibliography)
            source_stts = []
            
            # Tải registry để lấy tên file gốc
            try:
                with open(self.source_file, 'r', encoding='utf-8') as f:
                    registry = json.load(f).get(self.session_id, {})
            except Exception:
                registry = {}

            for meta in raw_sources_meta:
                sid = str(meta.get("source_id"))
                # locator bây giờ là toàn bộ object metadata thu được từ chunk
                current_locator_obj = meta 
                
                # Lấy tên file thật (nquylbkaeexr, v.v.)
                real_name = registry.get(sid, self.session_id)

                # Đối chiếu object locator với array_bibliography
                found_id = None
                for bib in self.array_bibliography:
                    # So sánh object locator giúp đảm bảo tính duy nhất trên nhiều loại tài liệu
                    if bib["name"] == real_name and bib["locator"] == current_locator_obj:
                        found_id = bib["id"]
                        break
                
                if found_id is None:
                    found_id = len(self.array_bibliography) + 1
                    self.array_bibliography.append({
                        "id": found_id,
                        "name": real_name,
                        "locator": current_locator_obj # Lưu dưới dạng object
                    })
                
                if found_id not in source_stts:
                    source_stts.append(found_id)

            result_node["content"] = content
            result_node["source"] = source_stts
            result_node["type"] = "leaf"
        
        else:
            result_node["content"] = ""

        return result_node

    def write_bibliography(self):
        """Tạo chuỗi bibliography hiển thị cuối bài"""
        self.bibliography = "## TÀI LIỆU THAM KHẢO\n"
        bib_lines = []
        for bib in self.array_bibliography:
            # Đối với chuỗi hiển thị, chúng ta chỉ cần in locator một cách ngắn gọn
            # nhưng trong array_bibliography vẫn lưu đầy đủ object.
            loc = bib['locator']
            loc_str = f"Block: {loc.get('block_index')}" if 'block_index' in loc else str(loc)
            bib_lines.append(f"[{bib['id']}] {bib['name']} ({loc_str})")
        
        if not bib_lines:
            self.bibliography += "Chưa có tài liệu tham khảo."
        else:
            self.bibliography += "\n".join(bib_lines)

    def wiki_compose(self):
        print(f"--- Bắt đầu viết bài: {self.name} ---")
        self.full_content = f"# {self.name}\n\n"
        
        for node in self.template.structure:
            processed_node = self._dfs_traverse(node)
            self.array_content.append(processed_node)
            self.full_content += f"{processed_node['title']}\n{processed_node['content']}\n\n"

        self.write_bibliography()
        self.full_content += f"\n{self.bibliography}"
        
        print("--- Hoàn tất ---")
        return {
            "full_content": self.full_content,
            "array_content": self.array_content,
            "array_bibliography": self.array_bibliography
        }