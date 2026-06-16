from dotenv import load_dotenv
load_dotenv()
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-index1")

vector_store = PineconeVectorStore(embedding=embeddings, index=index)

print(vector_store.add_texts(["ring"]))

