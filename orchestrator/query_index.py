#!/usr/bin/env python3
"""
query_index.py — HoneyRAG retrieval + LLM pipeline (robust, debug-friendly)

Usage:
    python3 query_index.py "<query>" <k> "<llm_question>" [--timeout N] [--max-tokens M] [--model /path/to/model.gguf] [--stub]

Features:
- Loads FAISS index + metadata and returns top-k retrievals.
- Builds a SOC-analyst prompt and runs a local GPT4All model in a separate process with a timeout.
- Robust error handling, debug logging to stderr, and optional stub mode for testing.
- Prints only the LLM answer to stdout (so callers can capture it) and writes a record to answers.log.
- Forces CPU mode by default to avoid CUDA dependency issues; you can override via environment variables.
- Produces small debug files (llm_child_out.txt / llm_child_err.txt / llm_debug.log) when generation runs, to help troubleshooting.

Notes:
- Install dependencies: pip install sentence-transformers faiss-cpu gpt4all
- If your model file is large or your machine is low on RAM, generation on CPU can be slow; increase --timeout accordingly.
"""

from __future__ import annotations
import os
import sys
import time
import json
import signal
import argparse
from pathlib import Path
from multiprocessing import Process, Queue
from typing import Optional, List, Dict
from datetime import datetime, timezone

# Force CPU by default (safe)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("GGML_USE_CUDA", "0")

START = time.time()

# ---------------- Config / Paths ----------------
IDX_DIR = Path("faiss_index")
INDEX_FILE = IDX_DIR / "index.faiss"
META_FILE = IDX_DIR / "metadata.jsonl"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

DEFAULT_LOCAL_GGUF = Path.home() / ".gpt4all" / "phi-2.Q4_K_M.gguf"
DEFAULT_REGISTRY_MODEL = "gpt4all-falcon"

CHECKPOINT_FILE = Path("honeyrag.checkpoint.json")
PID_FILE = Path("honeyrag.pid")
ANS_FILE = Path("answers.log")

LLM_CHILD_OUT = Path("llm_child_out.txt")
LLM_CHILD_ERR = Path("llm_child_err.txt")
LLM_DEBUG_LOG = Path("llm_debug.log")

# ---------------- Utilities ----------------
def eprint(*args, **kwargs):
    """Print to stderr safely."""
    try:
        print(*args, file=sys.stderr, **kwargs)
    except Exception:
        pass

def safe_write(path: Path, text: str):
    try:
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass

def save_checkpoint(stage: str, info: Optional[dict] = None):
    try:
        CHECKPOINT_FILE.write_text(json.dumps({"stage": stage, "info": info or {}, "ts": time.time()}))
    except Exception:
        pass

def write_pid():
    try:
        PID_FILE.write_text(str(os.getpid()))
    except Exception:
        pass

def remove_pid():
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass

# Minimal safe signal handler
def _signal_handler(signum, frame):
    try:
        save_checkpoint("stopped", {"signal": int(signum)})
    except Exception:
        pass
    try:
        remove_pid()
    except Exception:
        pass
    os._exit(0)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
write_pid()

# ---------------- Argument parsing ----------------
parser = argparse.ArgumentParser(prog="query_index.py", description="Retrieval + local LLM pipeline (HoneyRAG)")
parser.add_argument("query", nargs="?", default="failed ssh login", help="Search query text")
parser.add_argument("k", nargs="?", type=int, default=5, help="Number of retrievals (k)")
parser.add_argument("llm_question", nargs="?", default=None, help="Optional LLM question/prompt")
parser.add_argument("--timeout", type=int, default=120, help="LLM generation timeout (seconds)")
parser.add_argument("--max-tokens", type=int, default=120, help="Max tokens for generation")
parser.add_argument("--model", type=str, default=None, help="Path to local GGUF model (overrides default)")
parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="SentenceTransformer embed model")
parser.add_argument("--stub", action="store_true", help="Use a deterministic stub response (no real model)")
args = parser.parse_args()

# ---------------- Optional stub (for testing integration) ----------------
def stub_answer(query: str, context: str) -> str:
    now = datetime.now(timezone.utc).isoformat()
    answer = (
        "Summary: Suspicious activity detected: repeated failed SSH attempts and a wget of a suspicious URL.\n"
        "1) Containment: block the offending IPs at the firewall and isolate affected hosts.\n"
        "2) Investigation: collect auth logs, memory snapshot, and search for persistence indicators.\n"
        f"IOCs: ip:10.2.3.118; file:/tmp/malware.sh; time:{now}\n"
        "### END"
    )
    return answer

# ---------------- Load retrieval components (faiss + embedder) ----------------
EMBED_MODEL = args.embed_model
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception as e:
    eprint("[ERROR] Missing retrieval dependencies:", repr(e))
    raise

# Validate index and metadata
if not INDEX_FILE.exists() or not META_FILE.exists():
    eprint(f"[ERROR] Missing index or metadata. Expected {INDEX_FILE} and {META_FILE}.")
    remove_pid()
    sys.exit(2)

# Load embedder and index
try:
    eprint(f"[DEBUG] Loading embedder {EMBED_MODEL} ...")
    EMBEDDER = SentenceTransformer(EMBED_MODEL)
    eprint("[DEBUG] Embedder loaded")
except Exception as e:
    eprint("[ERROR] Failed to load embedder:", repr(e))
    raise

try:
    eprint(f"[DEBUG] Loading FAISS index from {INDEX_FILE} ...")
    INDEX = faiss.read_index(str(INDEX_FILE))
    eprint("[DEBUG] FAISS index loaded")
except Exception as e:
    eprint("[ERROR] Failed to load FAISS index:", repr(e))
    raise

try:
    with open(META_FILE, "r", encoding="utf-8") as fh:
        META = [json.loads(line) for line in fh]
    eprint(f"[DEBUG] Loaded metadata ({len(META)} entries)")
except Exception as e:
    eprint("[ERROR] Failed to load metadata:", repr(e))
    raise

# ---------------- Retrieval helpers ----------------
def search(query: str, k: int = 5) -> List[Dict]:
    q_emb = EMBEDDER.encode([query]).astype("float32")
    D, I = INDEX.search(q_emb, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({"score": float(dist), "index": int(idx), "metadata": META[idx]})
    return results

def format_context(results: List[Dict], max_items: int = 5) -> str:
    items = results[:max_items]
    lines = []
    for r in items:
        m = r["metadata"]
        ts = m.get("timestamp", "unknown-time")
        src = m.get("src_ip", m.get("ip", "unknown-ip"))
        msg = m.get("message") or m.get("event") or m.get("msg") or "<no message>"
        lines.append(f"[{ts}] {src} → {msg}")
    return "\n\n".join(lines)

def build_prompt(context: str, question: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "You are a SOC analyst. Provide the following in order:\n"
        "1) A concise summary (2–3 sentences) of suspicious activity or 'No suspicious activity' if none.\n"
        "2) Two numbered next steps, each with a short explanation (Containment and Investigation).\n"
        "3) A single-line IOCs list in this format: IOCs: ip:<ip>; file:<path>; time:<timestamp>\n"
        "End your response with the explicit marker '### END'.\n"
    )

# ---------------- LLM worker (child process) ----------------
def _llm_worker(q: Queue, model_path: str, prompt: str, max_tokens: int):
    """
    Child process: loads GPT4All model and generates text.
    Writes debug outputs to llm_child_out.txt / llm_child_err.txt for inspection.
    Puts {"ok": text} or {"err": repr(e)} into queue.
    """
    try:
        # Import inside child to avoid pickling issues and to isolate errors
        from gpt4all import GPT4All
    except Exception as e:
        err = f"Import gpt4all failed: {repr(e)}"
        safe_write(LLM_CHILD_ERR, err)
        try:
            q.put({"err": err})
        except Exception:
            pass
        return

    try:
        # instantiate model
        g = GPT4All(model_path)
        # prefer chat_session if available
        if hasattr(g, "chat_session"):
            with g.chat_session() as session:
                resp = session.generate(prompt, max_tokens=max_tokens)
        else:
            resp = g.generate(prompt, max_tokens=max_tokens)

        # Normalize response to string
        if resp is None:
            out_text = ""
        elif isinstance(resp, str):
            out_text = resp
        elif isinstance(resp, dict):
            out_text = resp.get("text") or resp.get("response") or str(resp)
        elif isinstance(resp, (list, tuple)):
            out_text = "\n".join(str(x) for x in resp)
        else:
            out_text = str(resp)

        # Write child output for debugging
        safe_write(LLM_CHILD_OUT, out_text or "")
        try:
            q.put({"ok": out_text})
        except Exception:
            pass
    except Exception as e:
        # Write exception to child err file
        safe_write(LLM_CHILD_ERR, repr(e))
        try:
            q.put({"err": repr(e)})
        except Exception:
            pass

def ask_llm_with_timeout(question: str, context: str, model_arg: str, max_tokens: int, timeout: int) -> str:
    """
    Spawn child process to run LLM. Return generated text (trimmed to before ### END if present).
    Raises TimeoutError on timeout, RuntimeError on child error.
    """
    prompt = build_prompt(context, question)
    safe_write(LLM_DEBUG_LOG, f"START {datetime.utcnow().isoformat()} model={model_arg} timeout={timeout}\n")
    q = Queue()
    p = Process(target=_llm_worker, args=(q, model_arg, prompt, max_tokens))
    p.start()
    p.join(timeout)
    if p.is_alive():
        eprint(f"[WARN] LLM worker exceeded timeout ({timeout}s), terminating")
        p.terminate()
        p.join(5)
        # try to get any queued message
        try:
            result = q.get_nowait()
        except Exception:
            result = None
        if result and "ok" in result:
            text = result["ok"]
            if isinstance(text, str) and "### END" in text:
                return text.split("### END")[0].strip()
            return text
        raise TimeoutError(f"LLM generation timed out after {timeout} seconds")

    # child finished
    try:
        result = q.get_nowait()
    except Exception:
        raise RuntimeError("LLM worker finished but returned no result")

    if "ok" in result:
        text = result["ok"]
        if isinstance(text, str) and "### END" in text:
            return text.split("### END")[0].strip()
        return text
    raise RuntimeError(f"LLM worker error: {result.get('err')}")

# ---------------- Main flow ----------------
def main():
    query = args.query
    k = args.k
    llm_question = args.llm_question
    timeout = args.timeout
    max_tokens = args.max_tokens
    model_path = args.model if args.model else (str(DEFAULT_LOCAL_GGUF) if DEFAULT_LOCAL_GGUF.exists() else DEFAULT_REGISTRY_MODEL)

    eprint(f"[DEBUG] Query: {query!r} k={k} llm_question={'<present>' if llm_question else '<none>'}")
    eprint(f"[DEBUG] Model arg: {model_path}  timeout={timeout} max_tokens={max_tokens}  stub={args.stub}")

    # Retrieval
    try:
        results = search(query, k=k)
    except Exception as e:
        eprint("[ERROR] Retrieval failed:", repr(e))
        remove_pid()
        sys.exit(3)

    # Print debug retrieval info to stderr
    eprint("[DEBUG] Top matches (score, index, metadata snippet):")
    for r in results:
        m = r["metadata"]
        snippet = (m.get("message") or m.get("event") or m.get("msg") or "")[:200]
        eprint(f"  idx={r['index']} score={r['score']:.6f} snippet={snippet!r}")

    ctx = format_context(results, max_items=min(5, k))
    eprint("[DEBUG] Context built (first 500 chars):")
    eprint(ctx[:500])

    save_checkpoint("retrieval_done", {"query": query, "k": k, "context_snippet": ctx[:200]})

    if not llm_question:
        # Nothing to call LLM for; exit after printing context to stderr
        eprint("[INFO] No LLM question provided; exiting.")
        remove_pid()
        sys.exit(0)

    # If stub mode requested, return deterministic answer quickly
    if args.stub:
        ans = stub_answer(query, ctx)
        print(ans)
        # append to answers.log
        try:
            ANS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(ANS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "query": query, "context": ctx, "answer": ans}) + "\n")
        except Exception:
            pass
        remove_pid()
        return

    # Try to run LLM. If model path points to a local file that doesn't exist, fallback to registry name.
    if model_path != DEFAULT_REGISTRY_MODEL and not Path(model_path).exists():
        eprint(f"[WARN] Specified model path {model_path} not found; falling back to registry model {DEFAULT_REGISTRY_MODEL}")
        model_path = DEFAULT_REGISTRY_MODEL

    # Attempt generation with retries and helpful debug logging
    try:
        answer = ask_llm_with_timeout(llm_question, ctx, model_arg=model_path, max_tokens=max_tokens, timeout=timeout)
    except TimeoutError as te:
        eprint("[ERROR] LLM timed out:", repr(te))
        # Try one more time with extended timeout (conservative)
        extended = max(timeout * 2, 300)
        eprint(f"[INFO] Retrying with extended timeout {extended}s ...")
        try:
            answer = ask_llm_with_timeout(llm_question, ctx, model_arg=model_path, max_tokens=max_tokens, timeout=extended)
        except Exception as e2:
            eprint("[ERROR] Retry failed:", repr(e2))
            save_checkpoint("llm_timeout", {"timeout": timeout})
            remove_pid()
            sys.exit(4)
    except Exception as e:
        eprint("[ERROR] LLM generation failed:", repr(e))
        save_checkpoint("llm_error", {"error": repr(e)})
        remove_pid()
        sys.exit(5)

    # Print answer to stdout for caller (dashboard/subprocess capture)
    if answer is None:
        answer = ""
    # Ensure answer ends with marker if not present (for consistency)
    if "### END" not in answer:
        answer = answer.strip() + "\n### END"

    # Output only the answer to stdout
    print(answer)

    # Append to answers.log for monitoring (best-effort)
    try:
        ANS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ANS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "query": query, "context": ctx, "answer": answer}) + "\n")
    except Exception as e:
        eprint("[WARN] Failed to append to answers.log:", repr(e))

    save_checkpoint("llm_done", {"answer_snippet": (answer or "")[:200]})
    remove_pid()
    eprint(f"[DEBUG] TOTAL runtime {time.time()-START:.2f}s")

if __name__ == "__main__":
    main()
