import os
import warnings
import logging
import time
import json
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import re

# Set up cache directory for HuggingFace models
cache_dir = os.path.join(os.getcwd(), ".cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_LOGGING_LEVEL'] = 'ERROR'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'

warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pdf_parser import parse_pdf_from_url_multithreaded as parse_pdf_from_url, parse_pdf_from_file_multithreaded as parse_pdf_from_file
from embedder import build_faiss_index, preload_model
from retriever import retrieve_chunks
from llm import query_gemini
import uvicorn

app = FastAPI(title="HackRx Insurance Policy Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Starting up HackRx Insurance Policy Assistant...")
    print("Preloading sentence transformer model...")
    preload_model()
    print("Model preloading completed. API is ready to serve requests!")

@app.get("/")
async def root():
    return {"message": "HackRx Insurance Policy Assistant API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

class LocalQueryRequest(BaseModel):
    document_path: str
    questions: list[str]

def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.replace("Bearer ", "")
    if not token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

def process_batch(batch_questions, context_chunks):
    return query_gemini(batch_questions, context_chunks)

def get_document_id_from_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def question_has_https_link(q: str) -> bool:
    return bool(re.search(r"https://[^\s]+", q))

# Document cache with thread safety
doc_cache = {}
doc_cache_lock = Lock()

# ----------------- CACHE CLEAR ENDPOINT -----------------
@app.delete("/api/v1/cache/clear")
async def clear_cache(doc_id: str = Query(None, description="Optional document ID to clear"),
                      url: str = Query(None, description="Optional document URL to clear"),
                      doc_only: bool = Query(False, description="If true, only clear document cache")):
    """
    Clear cache data.
    - No params: Clears ALL caches.
    - doc_id: Clears caches for that document only.
    - url: Same as doc_id but computed automatically from URL.
    - doc_only: Clears only document cache.
    """
    cleared = {}

    # If URL is provided, convert to doc_id
    if url:
        doc_id = get_document_id_from_url(url)

    if doc_id:
        if not doc_only:
            with doc_cache_lock:
                if doc_id in doc_cache:
                    del doc_cache[doc_id]
                    cleared["doc_cache"] = f"Cleared document {doc_id}"
    else:
        if not doc_only:
            with doc_cache_lock:
                doc_cache.clear()
                cleared["doc_cache"] = "Cleared ALL documents"

    return {"status": "success", "cleared": cleared}

@app.post("/api/v1/hackrx/run")
async def run_query(request: QueryRequest, token: str = Depends(verify_token)):
    start_time = time.time()
    timing_data = {}
    try:
        print("=== INPUT JSON ===")
        print(json.dumps({"documents": request.documents, "questions": request.questions}, indent=2))
        print("==================\n")

        print(f"Processing {len(request.questions)} questions...")

        # PDF Parsing and FAISS Caching (keep document caching for speed)
        doc_id = get_document_id_from_url(request.documents)
        with doc_cache_lock:
            if doc_id in doc_cache:
                print("✅ Using cached document...")
                cached = doc_cache[doc_id]
                text_chunks = cached["chunks"]
                index = cached["index"]
                texts = cached["texts"]
            else:
                print("⚙️ Parsing and indexing new document...")
                pdf_start = time.time()
                text_chunks = parse_pdf_from_url(request.documents)
                timing_data['pdf_parsing'] = round(time.time() - pdf_start, 2)

                index_start = time.time()
                index, texts = build_faiss_index(text_chunks)
                timing_data['faiss_index_building'] = round(time.time() - index_start, 2)

                doc_cache[doc_id] = {
                    "chunks": text_chunks,
                    "index": index,
                    "texts": texts
                }

        # Retrieve chunks for all questions — no QA caching
        retrieval_start = time.time()
        all_chunks = set()
        question_positions = {}
        for idx, question in enumerate(request.questions):
            top_chunks = retrieve_chunks(index, texts, question)
            all_chunks.update(top_chunks)
            question_positions.setdefault(question, []).append(idx)
        timing_data['chunk_retrieval'] = round(time.time() - retrieval_start, 2)
        print(f"Retrieved {len(all_chunks)} unique chunks for all questions")

        # Query Gemini LLM fresh for all questions
        context_chunks = list(all_chunks)
        batch_size = 10
        batches = [(i, request.questions[i:i + batch_size]) for i in range(0, len(request.questions), batch_size)]

        llm_start = time.time()
        results_dict = {}
        with ThreadPoolExecutor(max_workers=min(5, len(batches))) as executor:
            futures = [executor.submit(process_batch, batch, context_chunks) for _, batch in batches]
            for (start_idx, batch), future in zip(batches, futures):
                try:
                    result = future.result()
                    if isinstance(result, dict) and "answers" in result:
                        for j, answer in enumerate(result["answers"]):
                            results_dict[start_idx + j] = answer
                    else:
                        for j in range(len(batch)):
                            results_dict[start_idx + j] = "Error in response"
                except Exception as e:
                    for j in range(len(batch)):
                        results_dict[start_idx + j] = f"Error: {str(e)}"
        timing_data['llm_processing'] = round(time.time() - llm_start, 2)

        responses = [results_dict.get(i, "Not Found") for i in range(len(request.questions))]
        timing_data['total_time'] = round(time.time() - start_time, 2)

        print(f"\n=== TIMING BREAKDOWN ===")
        for k, v in timing_data.items():
            print(f"{k}: {v}s")
        print(f"=======================\n")

        print(f"=== OUTPUT JSON ===")
        print(json.dumps({"answers": responses}, indent=2))
        print(f"==================\n")

        return {"answers": responses}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/hackrx/local")
async def run_local_query(request: LocalQueryRequest):
    start_time = time.time()
    timing_data = {}
    try:
        print("=== INPUT JSON ===")
        print(json.dumps({"document_path": request.document_path, "questions": request.questions}, indent=2))
        print("==================\n")

        print(f"Processing {len(request.questions)} questions locally...")

        pdf_start = time.time()
        text_chunks = parse_pdf_from_file(request.document_path)
        timing_data['pdf_parsing'] = round(time.time() - pdf_start, 2)
        print(f"Extracted {len(text_chunks)} text chunks from PDF")

        index_start = time.time()
        index, texts = build_faiss_index(text_chunks)
        timing_data['faiss_index_building'] = round(time.time() - index_start, 2)

        retrieval_start = time.time()
        all_chunks = set()
        for question in request.questions:
            top_chunks = retrieve_chunks(index, texts, question)
            all_chunks.update(top_chunks)
        timing_data['chunk_retrieval'] = round(time.time() - retrieval_start, 2)
        print(f"Retrieved {len(all_chunks)} unique chunks")

        questions = request.questions
        context_chunks = list(all_chunks)
        batch_size = 20
        batches = [(i, questions[i:i + batch_size]) for i in range(0, len(questions), batch_size)]

        llm_start = time.time()
        results_dict = {}
        with ThreadPoolExecutor(max_workers=min(5, len(batches))) as executor:
            futures = [executor.submit(process_batch, batch, context_chunks) for _, batch in batches]
            for (start_idx, batch), future in zip(batches, futures):
                try:
                    result = future.result()
                    if isinstance(result, dict) and "answers" in result:
                        for j, answer in enumerate(result["answers"]):
                            results_dict[start_idx + j] = answer
                    else:
                        for j in range(len(batch)):
                            results_dict[start_idx + j] = "Error in response"
                except Exception as e:
                    for j in range(len(batch)):
                        results_dict[start_idx + j] = f"Error: {str(e)}"
        timing_data['llm_processing'] = round(time.time() - llm_start, 2)

        responses = [results_dict.get(i, "Not Found") for i in range(len(questions))]
        timing_data['total_time'] = round(time.time() - start_time, 2)

        print(f"\n=== TIMING BREAKDOWN ===")
        for k, v in timing_data.items():
            print(f"{k}: {v}s")
        print(f"=======================\n")

        print(f"=== OUTPUT JSON ===")
        print(json.dumps({"answers": responses}, indent=2))
        print(f"==================\n")

        return {"answers": responses}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
