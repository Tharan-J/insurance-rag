import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

cache_dir = os.path.join(os.getcwd(), ".cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

_model = None

def preload_model(model_name="paraphrase-MiniLM-L3-v2"):
    global _model
    if _model is not None:
        return _model

    print(f"Preloading sentence transformer model: {model_name}...")
    try:
        _model = SentenceTransformer(model_name, cache_folder=cache_dir)
    except Exception as e:
        print(f"Primary model load failed: {e}")
        fallback_name = "sentence-transformers/" + model_name
        print(f"Trying fallback: {fallback_name}")
        _model = SentenceTransformer(fallback_name, cache_folder=cache_dir)

    print("✅ Model ready.")
    return _model

def get_model():
    return preload_model()

def build_faiss_index(chunks, batch_size=128, show_progress_bar=False):
    model = get_model()
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks
