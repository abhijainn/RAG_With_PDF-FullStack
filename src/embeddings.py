from langchain_community.embeddings import HuggingFaceEmbeddings

def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
