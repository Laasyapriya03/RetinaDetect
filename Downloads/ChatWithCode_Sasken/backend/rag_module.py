import os
import json
import uuid
import re 
import tempfile # For handling temporary files
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions 
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Load model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
# Define the embedding function for ChromaDB
EMBEDDING_FUNCTION_CHROMA = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')


# Define default parameters for retrieval
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.7

COLLECTION_NAME = "code_chunks"

# Setup ChromaDB client (initialized in main.py's lifespan event)
client = None
collection = None 

# --- Start of extract_code_chunks function ---
def extract_code_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts semantically meaningful chunks from C/C++ code,
    prioritizing function bodies and including comments.
    Includes function names and type (code/comment) as metadata.
    """
    chunks = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    lines = content.splitlines()
    line_num = 0

    while line_num < len(lines):
        line = lines[line_num].strip()
        
        # --- Handle Multi-line Comments (Block Comments) ---
        if line.startswith('/*'):
            comment_start_line = line_num + 1
            comment_content_lines = [line]
            line_num += 1 # Move to the next line after '/*'
            while line_num < len(lines) and not lines[line_num].strip().endswith('*/'):
                comment_content_lines.append(lines[line_num])
                line_num += 1
            if line_num < len(lines): # Include the line with '*/'
                comment_content_lines.append(lines[line_num])
            
            chunks.append({
                "content": "\n".join(comment_content_lines),
                "source": file_path, # Use original file_path (temp file path)
                "start_line": comment_start_line,
                "type": "comment",
                "function_name": None
            })
            line_num += 1 # Move past the '*/' line
            continue

        # --- Handle Single-line Comments ---
        if line.startswith('//'):
            chunks.append({
                "content": line,
                "source": file_path, # Use original file_path (temp file path)
                "start_line": line_num + 1,
                "type": "comment",
                "function_name": None
            })
            line_num += 1
            continue
        
        # --- Attempt to find function definitions (more robustly with brace counting) ---
        function_signature_pattern = re.compile(r'^\s*(?:[\w\d_]+\s+)*([\w_][\w\d_]*)\s*\([^;]*\)\s*(?:const|noexcept)?\s*({)?\s*$')
        
        match = function_signature_pattern.match(line)
        
        if match and ';' not in line: # Exclude declarations ending with semicolon
            potential_func_name = match.group(1)
            
            func_start_line_idx = line_num
            brace_count = 0
            function_body_lines = []
            
            temp_line_idx = line_num
            found_opening_brace = False
            while temp_line_idx < len(lines) and (temp_line_idx - line_num) < 10: # Look ahead a few lines
                current_temp_line = lines[temp_line_idx]
                function_body_lines.append(current_temp_line)
                if '{' in current_temp_line:
                    brace_count += current_temp_line.count('{')
                    brace_count -= current_temp_line.count('}') # Handle { } on same line
                    found_opening_brace = True
                    break
                temp_line_idx += 1

            if found_opening_brace and brace_count > 0:
                line_num = temp_line_idx + 1 # Start from the line after the opening brace
                
                while line_num < len(lines) and brace_count > 0:
                    current_body_line = lines[line_num]
                    function_body_lines.append(current_body_line)
                    brace_count += current_body_line.count('{')
                    brace_count -= current_body_line.count('}')
                    line_num += 1
                
                if brace_count == 0 and len(function_body_lines) > 0:
                    chunks.append({
                        "content": "\n".join(function_body_lines),
                        "source": file_path, # Use original file_path (temp file path)
                        "start_line": func_start_line_idx + 1,
                        "type": "code",
                        "function_name": potential_func_name
                    })
                    continue 
                elif brace_count > 0 and line_num >= len(lines):
                    if len(function_body_lines) > 0:
                         chunks.append({
                            "content": "\n".join(function_body_lines),
                            "source": file_path, # Use original file_path (temp file path)
                            "start_line": func_start_line_idx + 1,
                            "type": "code",
                            "function_name": potential_func_name
                        })
                    continue
        
        # --- Handle remaining code lines (as generic code chunks) ---
        if line.strip(): 
            chunks.append({
                "content": line,
                "source": file_path, # Use original file_path (temp file path)
                "start_line": line_num + 1,
                "type": "code",
                "function_name": None 
            })
        line_num += 1
        
    return chunks
# --- End of extract_code_chunks function ---

# --- NEW FUNCTION FOR UPLOADED FILES ---
def process_and_store_uploaded_file(file_content_bytes: bytes, original_filename: str):
    """
    Processes a single uploaded file's content and stores its chunks in ChromaDB.
    """
    global collection 
    if collection is None:
        print("Error: ChromaDB collection not initialized. Cannot process uploaded file.")
        raise RuntimeError("ChromaDB collection not initialized.")

    # Create a temporary file to write the content
    # This allows extract_code_chunks to read it as a regular file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=os.path.splitext(original_filename)[1]) as temp_f:
        temp_f.write(file_content_bytes)
        temp_file_path = temp_f.name
    
    try:
        print(f"Processing uploaded file: {original_filename} (temp path: {temp_file_path})...")
        chunks = extract_code_chunks(temp_file_path)

        ids_batch = []
        embeddings_batch = []
        documents_batch = []
        metadatas_batch = []
        batch_size = 100 

        for chunk in chunks:
            # Use original_filename in the source metadata for clarity in retrieval
            chunk_id = f"{original_filename}-{chunk['start_line']}-{abs(hash(chunk['content']))}"
            embedding = model.encode(chunk["content"]).tolist()

            metadata = {
                "source": original_filename, # Store original filename
                "start_line": int(chunk["start_line"]),
                "type": str(chunk.get("type", "code")),
                "function_name": str(chunk.get("function_name", ""))
            }
            
            ids_batch.append(chunk_id)
            embeddings_batch.append(embedding)
            documents_batch.append(chunk["content"])
            metadatas_batch.append(metadata)

            if len(ids_batch) >= batch_size:
                try:
                    collection.add(
                        ids=ids_batch,
                        embeddings=embeddings_batch,
                        documents=documents_batch,
                        metadatas=metadatas_batch
                    )
                    print(f"Added batch of {len(ids_batch)} chunks for {original_filename}.")
                except Exception as e:
                    print(f"Error adding batch to ChromaDB for {original_filename}: {e}")
                
                ids_batch = []
                embeddings_batch = []
                documents_batch = []
                metadatas_batch = []
        
        if ids_batch:
            try:
                collection.add(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    documents=documents_batch,
                    metadatas=metadatas_batch
                )
                print(f"Added final batch of {len(ids_batch)} chunks for {original_filename}.")
            except Exception as e:
                print(f"Error adding final batch to ChromaDB for {original_filename}: {e}")

        print(f"Finished processing {original_filename}. Total chunks added: {collection.count()}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

# --- REMOVED FUNCTIONS: get_code_files and process_and_store_local_code ---
# These functions are no longer needed as files are uploaded via the frontend.
# def get_code_files(directory): ...
# def process_and_store_local_code(): ...


def retrieve_relevant_chunks(query: str, top_k: int = DEFAULT_TOP_K, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> list[dict]:
    global collection 
    if collection is None:
        print("Error: ChromaDB collection not initialized. Cannot retrieve chunks.")
        return [] 

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )

    retrieved_info = []
    if results and results['documents'] and results['distances'] and results['metadatas']:
        for i in range(len(results['documents'][0])):
            doc_content = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]

            if distance < similarity_threshold:
                retrieved_info.append({
                    "content": doc_content,
                    "source": metadata.get("source", "N/A"),
                    "start_line": metadata.get("start_line", -1),
                    "type": metadata.get("type", "code"),
                    "function_name": metadata.get("function_name", "N/A"), 
                    "distance": distance
                })
            else:
                print(f"Skipping chunk due to low similarity (distance {distance:.4f} >= threshold {similarity_threshold:.4f}): {doc_content[:50]}...")

    retrieved_info.sort(key=lambda x: x['distance'])

    unique_retrieved_info = []
    seen_contents = set()
    for item in retrieved_info:
        if item['content'] not in seen_contents:
            unique_retrieved_info.append(item)
            seen_contents.add(item['content'])
            
    return unique_retrieved_info