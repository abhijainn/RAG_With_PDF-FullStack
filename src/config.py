import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # Astra / Cassandra
    ASTRA_TOKEN: str = os.getenv("ASTRA_TOKEN", "")
    ASTRA_SCB: str = os.getenv("ASTRA_SCB", "src\\data\\secure-connect-pdf-qna.zip")
    ASTRA_KEYSPACE: str = os.getenv("ASTRA_KEYSPACE", "pdf_qna_name")
    ASTRA_TABLE: str = os.getenv("ASTRA_TABLE", "pdf_qna_table")

    # Models / runtime
    LLM_NAME: str = os.getenv("LLM_NAME", "Qwen/Qwen2.5-3B-Instruct")
    DEVICE: str = os.getenv("DEVICE", "cuda")

settings = Settings()
print("ASTRA_SCB ->", settings.ASTRA_SCB)
print("Exists?   ->", os.path.exists(settings.ASTRA_SCB))
