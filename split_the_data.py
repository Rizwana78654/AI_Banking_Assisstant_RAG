from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load Credit Card file
credit_loader = TextLoader("./Data_Location/credit_card.txt")
credit_documents = credit_loader.load()

# Load Current Account file
current_loader = TextLoader("./Data_Location/current_account.txt")
current_documents = current_loader.load()

# Combine both documents
documents = credit_documents + current_documents

# Split documents
chunk_obj = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Create chunks
result = chunk_obj.split_documents(documents)

print(f"No of chunks [documents]: {len(result)}")

print(result[0].page_content)
print(result[1].page_content)