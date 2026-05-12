cat > build_index.py <<'PY' 
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

PARSED = Path("data/parsed_logs.jsonl")
EMB_DIR = Path("embeddings")
IDX_DIR = Path("faiss_index")
EMB_DIR.mkdir(parents=True, exist_ok=True)
IDX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)

def load_records(path):
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            recs.append(json.loads(line))
    return recs

def build_embeddings(recs, batch_size=64):
    texts = [r.get("message","") for r in recs]
    embs = MODEL.encode(texts, show_progress_bar=True, batch_size=batch_size)
    return np.array(embs).astype("float32")

def main():
    recs = load_records(PARSED)
    if not recs:
        print("No records found.")
        return
    emb = build_embeddings(recs)
    np.save(EMB_DIR / "embeddings.npy", emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    faiss.write_index(index, str(IDX_DIR / "index.faiss"))
    with open(IDX_DIR / "metadata.jsonl", "w", encoding="utf-8") as out:
        for r in recs:
            out.write(json.dumps({
                "timestamp": r.get("timestamp"),
                "src_ip": r.get("src_ip"),
                "message": r.get("message")
            }, ensure_ascii=False) + "\n")
    print("Index built:", IDX_DIR / "index.faiss")
    print("Metadata saved:", IDX_DIR / "metadata.jsonl")

if __name__ == "__main__":
    main()
PY
