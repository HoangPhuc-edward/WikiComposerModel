import os
import json
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from extractor import Extractor

class Preprocessor:
    def __init__(self, session_id, base_dir="data_storage", extract_model_size="base", embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", chunk_size=800, chunk_overlap=100):
        self.session_id = session_id
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw", session_id)
        self.vector_path = os.path.join(base_dir, "vector_db")
        self.source_file = os.path.join(base_dir, "source.json")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.vector_path, exist_ok=True)


        self.extractor = Extractor(model_size=extract_model_size)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        self.chroma_client = chromadb.PersistentClient(path=self.vector_path)
        self.collection = self.chroma_client.get_or_create_collection(name="wiki_docs")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

        self._init_source_registry()

    def _init_source_registry(self):
        data = {}
        if os.path.exists(self.source_file):
            with open(self.source_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        
        if self.session_id not in data:
            data[self.session_id] = {}
            with open(self.source_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _add_source(self, input_source):
        with open(self.source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session_data = data.get(self.session_id, {})
        
        for sid, name in session_data.items():
            if name == input_source:
                return int(sid)
        
        new_id = len(session_data) + 1
        data[self.session_id][str(new_id)] = input_source
        
        with open(self.source_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return new_id

    def extract(self, input_source, input_type):
        extracted_data = []
        if input_type == "url": 
            extracted_data = self.extractor.extract_website(input_source)
        elif input_type in ["pdf", "docx"]: 
            extracted_data = self.extractor.extract_text_file(input_source)
        elif input_type == "youtube": 
            extracted_data = self.extractor.extract_youtube(input_source) 
        elif input_type == "audio": 
            extracted_data = self.extractor.extract_mp3(input_source)
        elif input_type == "txt":
            extracted_data = self.extractor.extract_txt(input_source)



        if not extracted_data:
            return None, None

        source_id = self._add_source(input_source)
        safe_name = f"source_{source_id}.json"
        
        with open(os.path.join(self.raw_dir, safe_name), 'w', encoding='utf-8') as f:
            json.dump({"source": input_source, "content": extracted_data}, f, ensure_ascii=False)
            
        return extracted_data, source_id

    def chunking(self, extracted_data):
        final_chunks = []
        
        for item in extracted_data:
            text = item.get("text", "")
            meta = item.get("metadata", {})
            
            if len(text) > self.text_splitter._chunk_size:
                sub_chunks = self.text_splitter.split_text(text)
                for sub in sub_chunks:
                    final_chunks.append({"text": sub, "metadata": meta})
            else:
                final_chunks.append({"text": text, "metadata": meta})
                
        return final_chunks

    def save(self, chunks, source_id):
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            documents.append(chunk["text"])
            
            meta = {
                "doc_name": self.session_id,
                "source_id": source_id,
                "chunk_index": i
            }
            
            source_meta = chunk.get("metadata", {})
            meta["source_type"] = source_meta.get("source_type", "unknown")
            locator = source_meta.get("locator", {})
            meta.update(locator)
            
            metadatas.append(meta)

        embeddings = self.embedding_model.encode(documents).tolist()
        self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
        return True

    def execute(self, input_source, input_type):
        data, source_id = self.extract(input_source, input_type)
        if not data:
            return False
            
        chunks = self.chunking(data)
        return self.save(chunks, source_id)




