import argparse
from pathlib import Path

from src.app.config import settings
from src.app.embeddings import build_embeddings
from src.app.llm import build_llm
from src.app.ingest import build_index
from src.app.vectorstore import connect_astra

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--question", required=True)
    args = ap.parse_args()

    if not settings.ASTRA_TOKEN:
        raise SystemExit("Set APPLICATION_TOKEN in your environment (AstraCS:...)")

    if not Path(settings.ASTRA_SCB).exists():
        raise SystemExit(f"SCB not found: {settings.ASTRA_SCB}")

    if not Path(args.pdf).exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    llm = build_llm(settings.LLM_NAME)
    embeddings = build_embeddings()
    cluster, session = connect_astra(settings.ASTRA_SCB, settings.APPLICATION_TOKEN)

    try:
        index = build_index(args.pdf, session, settings.ASTRA_KEYSPACE, settings.ASTRA_TABLE, embeddings)
        resp = index.query_with_sources(args.question, llm=llm)
        print("\nANSWER:\n", resp.get("answer", resp))
        print("\nSOURCES:\n", resp.get("sources", ""))
    finally:
        cluster.shutdown()

if __name__ == "__main__":
    main()
