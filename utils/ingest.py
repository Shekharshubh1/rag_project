import os
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.embeddings import HuggingFaceEmbeddings

def ingest_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    PineconeStore.from_documents(chunks, embeddings, index_name=index_name)
