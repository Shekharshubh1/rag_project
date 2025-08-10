import os
from huggingface_hub import InferenceClient
import pinecone
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.embeddings import HuggingFaceEmbeddings

def query_rag(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    vectorstore = PineconeStore.from_existing_index(index_name, embeddings)

    results = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])

    client = InferenceClient(model="microsoft/phi-3-mini-4k-instruct", token=os.getenv("HF_TOKEN"))
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = client.text_generation(prompt, max_new_tokens=300)
    return response
