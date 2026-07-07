import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

app = Flask(__name__)

# -----------------------------------
# Load Embeddings
# -----------------------------------
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# -----------------------------------
# Load FAISS Database
# -----------------------------------
vectorstore = FAISS.load_local(
    "DB",
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------------
# Load LLM
# -----------------------------------
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# -----------------------------------
# AI Function
# -----------------------------------
def safe_rag_response(topic, question):

    if not topic:
        return "Please select a banking service."

    if not question.strip():
        return "Please enter your question."

    try:

        # Search documents
        search_query = f"{topic}\n{question}"

        docs = vectorstore.similarity_search(
            search_query,
            k=3
        )

        if len(docs) == 0:
            return "Sorry, I couldn't find that information."

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        print("=" * 60)
        print("Selected Service:", topic)
        print("Question:", question)
        print("Documents Found:", len(docs))
        print(context)
        print("=" * 60)

        prompt = f"""
You are an SBI AI Banking Assistant.

Selected Banking Service:
{topic}

Context:
{context}

Customer Question:
{question}

Instructions:
- Answer ONLY from the given context.
- If the answer is not found in the context, reply:
Sorry, I couldn't find that information.
- Keep the answer short.
- Maximum 3 lines.
"""

        response = llm.invoke(prompt)

        return response.content.strip()

    except Exception as e:
        print("ERROR:", e)
        return "Something went wrong."


# -----------------------------------
# Home
# -----------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    answer = ""

    if request.method == "POST":

        topic = request.form.get("topic", "")
        question = request.form.get("question", "")

        answer = safe_rag_response(
            topic,
            question
        )

    return render_template(
        "index.html",
        answer=answer
    )


# -----------------------------------
# Run
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)