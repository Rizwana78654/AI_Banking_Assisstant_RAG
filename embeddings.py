import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# -----------------------------------
# Load Credit Card Dataset
# -----------------------------------
credit_loader = TextLoader(
    "./Data_Location/credit_card.txt",
    encoding="utf-8"
)

credit_documents = credit_loader.load()

# -----------------------------------
# Load Current Account Dataset
# -----------------------------------
current_loader = TextLoader(
    "./Data_Location/current_account.txt",
    encoding="utf-8"
)

current_documents = current_loader.load()

# -----------------------------------
# Combine both datasets
# -----------------------------------
documents = credit_documents + current_documents

# -----------------------------------
# Split into chunks
# -----------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("No of chunks:", len(chunks))

# -----------------------------------
# Ollama Embeddings
# -----------------------------------
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

# -----------------------------------
# Create FAISS Vector Store
# -----------------------------------
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

# -----------------------------------
# Save Vector Database
# -----------------------------------
vectorstore.save_local("DB")

print("✅ FAISS Vector Database created successfully!")