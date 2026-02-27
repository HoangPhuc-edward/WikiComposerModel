import os
import json
import re
import torch
from sentence_transformers import CrossEncoder

class WikiEvaluation:
    def __init__(self, article_array, source_dir_path, session_id, array_bibliography, nli_model=None):
        """
        article_array: Dữ liệu array của bài viết.
        source_dir_path: Thư mục 'data_storage/raw'.
        session_id: ID phiên làm việc.
        array_bibliography: Dữ liệu từ file _bibliography.json (BẮT BUỘC).
        """
        self.article_array = article_array
        self.source_dir_path = source_dir_path
        self.session_id = session_id
        self.array_bibliography = array_bibliography
        
        self.full_content = []
        self.source_map = {} 
        
        if nli_model:
            self.nli_model = nli_model
        else:
            print("--- Đang tải model NLI... ---")
            self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

    
    def preprocess(self):
        """
        Thực hiện tiền xử lý dữ liệu bài viết và nguồn.
        """
        # --- BƯỚC 1: XỬ LÝ BÀI VIẾT (DFS) ---
        self.full_content = [] # Reset
        # Bắt đầu duyệt từ root
        self._dfs_parse_article(self.article_array)
        
        # --- BƯỚC 2: XỬ LÝ NGUỒN ---
        all_cited_ids = set()
        for item in self.full_content:
            all_cited_ids.update(item['source_ids'])
            
        self._load_sources(all_cited_ids)
        
        print(f"✅ Preprocess hoàn tất: {len(self.full_content)} đoạn văn tách biệt.")

    def _dfs_parse_article(self, current_node):
        """
        Duyệt cây bài viết theo phương pháp DFS.
        Logic: 
        - Nếu mảng 'children' có ít nhất 1 phần tử -> Nút cha (Section): Duyệt tiếp các con.
        - Nếu mảng 'children' trống -> Nút lá (Leaf): Lấy nội dung và nguồn để đánh giá.
        """
        # Trường hợp input là một danh sách các nút (như mảng root ban đầu)
        if isinstance(current_node, list):
            for item in current_node:
                self._dfs_parse_article(item)
            return

        # Xử lý nút dạng Dictionary
        if isinstance(current_node, dict):
            children = current_node.get('children', [])
            
            if len(children) > 0:
                # ĐÂY LÀ NÚT CHA: Tiếp tục đệ quy xuống các nút con
                self._dfs_parse_article(children)
            else:
                # ĐÂY LÀ NÚT LÁ: Trích xuất nội dung và danh sách nguồn trực tiếp
                content = current_node.get('content', "")
                source_ids = current_node.get('source', [])
                
                # Gọi hàm xử lý text để đưa vào hàng chờ đánh giá
                self._process_text_segment(content, source_ids)

    def _process_text_segment(self, raw_text, source_ids):
        """
        Làm sạch văn bản và lưu trữ cùng với danh sách nguồn tương ứng.
        Vì 'source' đã được tách riêng, không cần dùng Regex để bóc tách từ text nữa.
        """
        if not raw_text or not raw_text.strip():
            return

        # Làm sạch các ký tự format rác (nếu còn sót lại từ LLM)
        # Loại bỏ các dấu *, # và khoảng trắng thừa
        clean_text = re.sub(r'[\*#]', '', raw_text)
        clean_text = clean_text.strip()

        if clean_text:
            # Lưu vào full_content để các hàm calculate_avg sử dụng
            self.full_content.append({
                'content': clean_text,
                'source_ids': source_ids  # source_ids giờ là mảng [1, 2, 3...] lấy trực tiếp từ JSON
            })

    def _load_sources(self, source_ids):
        """
        Đọc phân đoạn văn bản. 
        Xử lý đặc biệt cho 'web' (lấy full) và các loại khác (so khớp locator).
        """
        self.source_map = {}
        base_path = os.path.join(self.source_dir_path, self.session_id)
        bib_lookup = {item['id']: item for item in self.array_bibliography}

        for sid in source_ids:
            bib_entry = bib_lookup.get(sid)
            if not bib_entry: continue
            
            locator_info = bib_entry.get('locator', {})
            source_type = locator_info.get('source_type')
            real_source_id = locator_info.get('source_id') 
            file_path = os.path.join(base_path, f"source_{real_source_id}.json")
            
            extracted_text = ""
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)
                    
                    raw_chunks = raw_data.get('content', [])
                    if not isinstance(raw_chunks, list):
                        raw_chunks = [raw_data]

                    # --- LOGIC TRÍCH XUẤT THÔNG MINH ---
                    if source_type == "web" or source_type == "txt":
                        # Với Web, locator rỗng nên ta lấy ngay chunk đầu tiên (thường chỉ có 1)
                        if raw_chunks:
                            extracted_text = raw_chunks[0].get('text', "")
                    else:
                        # Với docx, pdf, video... thực hiện so khớp locator như cũ
                        for chunk in raw_chunks:
                            raw_meta = chunk.get('metadata', {})
                            raw_loc = raw_meta.get('locator', {})
                            
                            match = False
                            # Ép kiểu string để tránh lệch pha dữ liệu 0 vs "0"
                            if source_type == "docx":
                                if str(raw_loc.get("block_index")) == str(locator_info.get("block_index")):
                                    match = True
                            elif source_type == "pdf":
                                if str(raw_loc.get("page_number")) == str(locator_info.get("page_number")):
                                    match = True
                            elif source_type in ["audio", "video", "youtube"]:
                                if str(raw_loc.get("start_seconds")) == str(locator_info.get("start_seconds")):
                                    match = True
                            
                            if match:
                                extracted_text = chunk.get('text', "")
                                break
                    
                self.source_map[sid] = extracted_text.strip()
                
                # Debug nếu vẫn không thấy text
                if not self.source_map[sid]:
                    print(f"⚠️ Cảnh báo: Trích dẫn [{sid}] (Type: {source_type}) không có nội dung text.")

            except Exception as e:
                print(f"❌ Lỗi load nguồn {sid}: {e}")
                self.source_map[sid] = ""

    def _check_entailment(self, premise, hypothesis):
        """Helper: Check NLI (Private)"""
        if not premise.strip() or not hypothesis.strip():
            return False
        # Predic: 0: Contradiction, 1: Entailment, 2: Neutral
        scores = self.nli_model.predict([(premise, hypothesis)])
        return scores[0].argmax() == 1

    def calculate_citation_recall(self, statement, source_ids):
        """
        Tính Recall cho 1 đoạn văn:
        Gộp tất cả nguồn -> Check xem có suy ra được statement không.
        Return: 0 hoặc 1
        """
        if not source_ids: return 0
        
        # Gom nội dung các nguồn
        cited_contents = [self.source_map.get(sid, "") for sid in source_ids]
        combined_premise = " ".join(cited_contents)

        is_entailed = self._check_entailment(combined_premise, statement)
        return 1 if is_entailed else 0

    def calculate_citation_precision(self, statement, source_ids):
        """
        Tính Precision cho 1 đoạn văn:
        Phát hiện nguồn thừa.
        Return: 0.0 -> 1.0
        """
        # Nếu Recall = 0 (Sai ngay từ đầu) -> Precision = 0
        if self.calculate_citation_recall(statement, source_ids) == 0:
            return 0.0

        valid_count = 0
        cited_contents = [self.source_map.get(sid, "") for sid in source_ids]
        
        for i, target_content in enumerate(cited_contents):
            # Điều kiện A: Bản thân nó KHÔNG chứng minh được
            cond_a = not self._check_entailment(target_content, statement)
            
            # Điều kiện B: Các thằng còn lại VẪN chứng minh được
            remaining = [c for j, c in enumerate(cited_contents) if j != i]
            combined_remaining = " ".join(remaining)
            
            if not remaining:
                cond_b = False # Bỏ nó ra là hết sạch -> Bắt buộc phải có nó
            else:
                cond_b = self._check_entailment(combined_remaining, statement)
            
            # Nếu (A đúng) VÀ (B đúng) => Nó là thừa
            if cond_a and cond_b:
                pass # Irrelevant
            else:
                valid_count += 1 # Relevant
                
        return valid_count / len(source_ids) if source_ids else 0


    def calculate_avg_citation_recall(self):
        """Tính trung bình Recall cho toàn bài (%)"""
        if not self.full_content: return 0.0
        
        total_score = 0
        count = 0
        
        print("\n--- Đang tính AVG Recall ---")
        for item in self.full_content:
            # Chỉ tính những câu CÓ trích dẫn (hoặc tùy logic bạn muốn tính cả câu không trích dẫn)
            if item['source_ids']: 
                score = self.calculate_citation_recall(item['content'], item['source_ids'])
                total_score += score
                count += 1
                # Debug log nhỏ
                # print(f"Sentence: {item['content'][:30]}... | Sources: {item['source_ids']} | Recall: {score}")
        
        if count == 0: return 0.0
        return (total_score / count) * 100

    def calculate_avg_citation_precision(self):
        """Tính trung bình Precision cho toàn bài (%)"""
        if not self.full_content: return 0.0
        
        total_score = 0
        count = 0
        
        print("\n--- Đang tính AVG Precision ---")
        for item in self.full_content:
            if item['source_ids']:
                score = self.calculate_citation_precision(item['content'], item['source_ids'])
                total_score += score
                count += 1
        
        if count == 0: return 0.0
        return (total_score / count) * 100

    def debug(self, path):
        """Lưu trữ dữ liệu tiền xử lý và nguồn đã load để kiểm tra logic trích dẫn"""
        debug_data = {
            "session_id": self.session_id,
            "full_content_parsed": self.full_content,
            "source_map_loaded": self.source_map,
            "bibliography_reference": self.array_bibliography
        }
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, ensure_ascii=False, indent=4)
        print(f"📄 [Debug] Đã lưu dữ liệu WikiEvaluation tại: {path}")