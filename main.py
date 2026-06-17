from dotenv import load_dotenv
load_dotenv()
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader("./fullstack_guide.pdf")
docs = loader.load()

for doc in docs:
    doc.page_content = doc.page_content.replace("\n", " ")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=[". ", " ", "\n\n"]  
)

chunks = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-index1")


index.delete(delete_all=True)


vector_store = PineconeVectorStore(embedding=embeddings, index=index)
 
vector_store.add_documents(chunks)

print(vector_store.similarity_search(query="version control with git", k=1))

