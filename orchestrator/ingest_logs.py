#!/usr/bin/env python3
import argparse
import json
import time
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Paths (adjust if needed)
LOG_DEFAULT = "/home/sam_honeyrag/cowrie-artifacts/demo_logs.json"
INDEX_DIR = Path("/home/sam_honeyrag/cowrie-artifacts/faiss_index")
INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "metadata.jsonl"

# Embedding model settings
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # model output dim for all-MiniLM-L6-v2

# Ensure index directory exists
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_or_create_index():
    if INDEX_PATH.exists():
        idx = faiss.read_index(str(INDEX_PATH))
    else:
        idx = faiss.IndexFlatL2(EMBED_DIM)
    return idx

def append_metadata(meta_obj):
    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta_obj, ensure_ascii=False) + "\n")

def process_event(event, embedder, index):
    # Only ingest command input events
    if event.get("eventid") != "cowrie.command.input":
        return False
    msg = event.get("message") or event.get("input") or ""
    if not msg:
        return False
    vec = embedder.encode([msg])
    vec = np.array(vec, dtype="float32")
    index.add(vec)
    meta = {
        "timestamp": event.get("timestamp"),
        "src_ip": event.get("src_ip"),
        "message": msg
    }
    append_metadata(meta)
    print("[INGESTED]", msg)
    return True

def batch_ingest(path, batch_size=64):
    embedder = SentenceTransformer(EMBED_MODEL)
    index = load_or_create_index()
    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except Exception:
                continue
            try:
                if process_event(event, embedder, index):
                    count += 1
            except Exception as e:
                print("[SKIP]", e)
    if count > 0:
        faiss.write_index(index, str(INDEX_PATH))
    print(f"[DONE] batch ingest complete ({count} items)")

def tail_ingest(path):
    embedder = SentenceTransformer(EMBED_MODEL)
    index = load_or_create_index()
    # Open file and seek to end
    with open(path, "r", encoding="utf-8") as fh:
        fh.seek(0, os.SEEK_END)
        while True:
            line = fh.readline()
            if not line:
                time.sleep(0.5)
                continue
            try:
                event = json.loads(line)
            except Exception:
                print("[TAIL] invalid json line, skipping")
                continue
            try:
                if process_event(event, embedder, index):
                    # persist index after each successful add (safe for demo)
                    faiss.write_index(index, str(INDEX_PATH))
            except Exception as e:
                print("[ERROR]", e)

def main():
    p = argparse.ArgumentParser(description="Ingest Cowrie JSON into FAISS (batch or tail).")
    p.add_argument("--source", "-s", default=LOG_DEFAULT, help="Path to source JSONL (cowrie.json or demo_logs.json)")
    p.add_argument("--mode", "-m", choices=["batch", "tail"], default="batch", help="Ingest mode")
    args = p.parse_args()

    if not Path(args.source).exists():
        print("[ERROR] source file not found:", args.source)
        return

    if args.mode == "batch":
        batch_ingest(args.source)
    else:
        print("[TAIL] starting tail ingest on", args.source)
        tail_ingest(args.source)

if __name__ == "__main__":
    main()
