from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import os
import chromadb
import shutil # For clearing the database directory
from typing import List, Optional # <--- ADDED Optional import here
from fastapi.middleware.cors import CORSMiddleware

# Ensure these imports are correct based on your file structure
from backend import rag_module
from backend import llm_module # Assuming you have an llm_module.py

# Define the path to your ChromaDB data
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "code_chunks" # Must match COLLECTION_NAME in rag_module.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize ChromaDB client and collection
    print("Initializing ChromaDB client...")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    rag_module.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    rag_module.collection = rag_module.client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=rag_module.EMBEDDING_FUNCTION_CHROMA # Pass the embedding function
    )
    
    print("Application startup complete. ChromaDB ready for operations.")
    yield
    # Shutdown: No specific cleanup needed for persistent ChromaDB

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    temperature: float = llm_module.DEFAULT_TEMPERATURE
    top_k: int = rag_module.DEFAULT_TOP_K
    similarity_threshold: float = rag_module.DEFAULT_SIMILARITY_THRESHOLD

class CodeChunk(BaseModel):
    content: str
    source: str
    start_line: int
    type: str
    distance: Optional[float] = None
    function_name: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    retrieved_context: List[CodeChunk]
    debug_info: dict = {}

@app.get("/")
async def read_root():
    return {"message": "Chat with Your Code API is running!"}

@app.post("/upload_code_file")
async def upload_code_file(file: UploadFile = File(...)):
    """
    Receives an uploaded code file, processes it, and stores its chunks in ChromaDB.
    """
    if not file.filename.endswith(('.c', '.cpp', '.h')):
        raise HTTPException(status_code=400, detail="Only .c, .cpp, and .h files are allowed.")
    
    try:
        file_content_bytes = await file.read()
        print(f"Received file: {file.filename}, size: {len(file_content_bytes)} bytes")
        
        # Process and store the uploaded file
        # The rag_module will handle creating temporary file and processing
        rag_module.process_and_store_uploaded_file(file_content_bytes, file.filename)
        
        return {"message": f"File '{file.filename}' processed and indexed successfully."}
    except Exception as e:
        print(f"Error processing uploaded file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

@app.post("/clear_codebase")
async def clear_codebase():
    """
    Clears all data from the ChromaDB collection.
    """
    try:
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
            print(f"Cleared ChromaDB at {CHROMA_DB_PATH}")
            
            # Re-initialize the collection after clearing
            rag_module.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            rag_module.collection = rag_module.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=rag_module.EMBEDDING_FUNCTION_CHROMA
            )
            print("ChromaDB re-initialized after clearing.")
        else:
            print("ChromaDB directory does not exist, no clearing needed.")
        
        return {"message": "ChromaDB codebase cleared successfully."}
    except Exception as e:
        print(f"Error clearing ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear codebase: {e}")


@app.post("/ask/", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        # Retrieve relevant code chunks
        retrieved_chunks = rag_module.retrieve_relevant_chunks(
            request.query, 
            top_k=request.top_k, 
            similarity_threshold=request.similarity_threshold
        )
        
        # Format context for the LLM
        context = ""
        for i, chunk in enumerate(retrieved_chunks):
            func_info = f" (Function: {chunk['function_name']})" if chunk.get('function_name') else ""
            context += f"### Chunk {i+1} (Source: {os.path.basename(chunk['source'])}, Line: {chunk['start_line']}, Type: {chunk['type']}{func_info})\n"
            context += f"{chunk['content']}\n\n"

        if not context:
            answer = "I cannot answer this question based on the provided code context. No relevant code chunks were found."
        else:
            # Generate answer using the LLM, passing the temperature from the request
            answer = llm_module.generate_answer(
                request.query, # Pass the query
                context,       # Pass the context
                temperature=request.temperature # Pass the temperature
            )

        # Prepare response with retrieved context and debug info
        response_chunks = [CodeChunk(**chunk) for chunk in retrieved_chunks]
        
        return QueryResponse(
            answer=answer,
            retrieved_context=response_chunks,
            debug_info={
                "retrieved_chunk_count": len(retrieved_chunks),
                "query_top_k": request.top_k,
                "query_similarity_threshold": request.similarity_threshold,
                "llm_temperature": request.temperature
            }
        )

    except Exception as e:
        print(f"An error occurred in /ask/: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)