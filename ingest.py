from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Cassandra

def build_index(pdf_path: str, session, keyspace: str, table: str, embeddings):
    loader = PyPDFLoader(pdf_path)
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=Cassandra,
        embedding=embeddings,
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30),
        vectorstore_kwargs={
            "session": session,
            "keyspace": keyspace,
            "table_name": table,
        },
    )
    return index_creator.from_loaders([loader])
