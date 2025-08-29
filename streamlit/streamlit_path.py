import io
import time
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import faiss
import fitz  
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline



st.set_page_config(page_title="Chat with PDF (OSS)", layout="wide")



def extract_pdf_text_blocks(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Return a list of {'page': int, 'text': str} from a PDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    blocks = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text and text.strip():
            blocks.append({"page": i + 1, "text": text.strip()})
    return blocks


def chunk_text(text: str, max_chars=1000, overlap=150) -> List[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + max_chars])
        i += max_chars - overlap
    return out


def chunk_blocks(blocks, max_chars=1000, overlap=150):
    chunks, meta = [], []
    for b in blocks:
        for ch in chunk_text(b["text"], max_chars=max_chars, overlap=overlap):
            chunks.append(ch)
            meta.append({"page": b["page"], "len": len(ch)})
    return chunks, meta



@st.cache_resource(show_spinner="Loading embedding model (BGE-small)…")
def get_embedder():
    # Fast + strong default, 384-dim, normalized cosine
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu")
    return model


@st.cache_resource(show_spinner="Loading reranker…")
def get_reranker():
    # Lightweight cross-encoder (optional but helpful)
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading LLM…")
def get_llm(model_choice: str):

    name = {
        "Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
        "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    }[model_choice]

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Keeping it CPU-friendly by default (no bitsandbytes here)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    return pipe


def embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size=64) -> np.ndarray:
    vecs = embedder.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def build_faiss_index(vecs: np.ndarray):
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine because vectors normalized
    index.add(vecs)
    return index


def rerank_hits(reranker: CrossEncoder, query: str, candidates: List[Tuple[int, str]], top_m=5):
    pairs = [[query, text] for (_, text) in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    return [(idx, text, float(score)) for ((idx, text), score) in ranked[:top_m]]


def answer_with_context(llm_pipe, question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_text = "\n\n".join([f"[{i+1}] (p.{c['page']}) {c['text'][:900]}" for i, c in enumerate(contexts)])
    system = (
        "You are a helpful assistant that answers strictly using the supplied context. "
        "Cite sources as [index] with page numbers. If the answer is not in the context, say you don't know."
    )
    prompt = f"{system}\n\nQuestion: {question}\n\nContext:\n{ctx_text}\n\nAnswer:"
    out = llm_pipe(prompt)[0]["generated_text"]
    return out.split("Answer:")[-1].strip()



st.title("📄 Chat with PDF — Open-Source")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("LLM", ["Phi-3.5-mini-instruct", "Qwen2.5-3B-Instruct"], index=0)
    k = st.slider("Top-K retrieved", 3, 20, 10)
    m = st.slider("Top-M reranked", 1, 10, 5)
    do_rerank = st.checkbox("Use reranker (better answers, a bit slower)", value=True)
    st.markdown("---")
    st.caption("Tip: Qwen is stronger but wants a GPU; Phi runs OK on CPU.")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.meta = None

if uploaded and st.button("Build index"):
    pdf_bytes = uploaded.read()

    with st.spinner("Parsing PDF…"):
        blocks = extract_pdf_text_blocks(pdf_bytes)
        chunks, meta = chunk_blocks(blocks, max_chars=1100, overlap=200)

    with st.spinner("Embedding and indexing…"):
        embedder = get_embedder()
        vecs = embed_texts(embedder, chunks)
        index = build_faiss_index(vecs)

    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.meta = meta
    st.success(f"Indexed {len(chunks)} chunks from {len(blocks)} pages.")

st.markdown("---")
q = st.text_input("Ask a question about the PDF…")

if st.session_state.index is None:
    st.info("Upload a PDF and click **Build index** to get started.")
else:
    if q:
        t0 = time.time()
        with st.spinner("Retrieving…"):
            embedder = get_embedder()
            qv = embed_texts(embedder, [q])[0:1]
            sims, ids = st.session_state.index.search(qv, k)
            cands = []
            for idx in ids[0]:
                text = st.session_state.chunks[idx]
                cands.append((idx, text))

            if do_rerank:
                reranker = get_reranker()
                ranked = rerank_hits(reranker, q, cands, top_m=m)
                ctx = []
                for (idx, text, score) in ranked:
                    md = st.session_state.meta[idx]
                    ctx.append({"text": text, "page": md["page"], "score": score, "id": idx})
            else:
                ctx = []
                for idx, text in cands[:m]:
                    md = st.session_state.meta[idx]
                    ctx.append({"text": text, "page": md["page"], "score": None, "id": idx})

        with st.spinner("Generating answer…"):
            llm = get_llm(model_choice)
            answer = answer_with_context(llm, q, ctx)

        elapsed = time.time() - t0

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Citations")
        for i, c in enumerate(ctx, 1):
            with st.expander(f"[{i}] Page {c['page']} • Chunk {c['id']}"):
                st.write(c["text"])

        st.caption(f"Done in {elapsed:.1f}s")
