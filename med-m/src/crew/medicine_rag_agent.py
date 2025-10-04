import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from output_pydantic import MedicineInfoOutput, MedicineComparisonOutput

load_dotenv()

# -------------------------
# LLM Setup
# -------------------------
llm = LLM(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini/gemini-1.5-flash",
)

# -------------------------
# Embeddings + FAISS Index
# -------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = "vectorstore/faiss_medical_index"

if os.path.exists(FAISS_INDEX_PATH):
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_documents(
        [Document(page_content="Paracetamol is used for fever and mild pain.")],
        embeddings
    )
    db.save_local(FAISS_INDEX_PATH)

# -------------------------
# Agent Config
# -------------------------
rag_agent = Agent(
    name="medicine_rag_agent",
    role="Medical Information Assistant",
    goal="Answer user questions about medicines and diseases using retrieval-augmented generation.",
    backstory="Retrieves relevant context from medical knowledge base (FAISS) and answers queries clearly.",
    llm=llm
)

# -------------------------
# Tasks
# -------------------------
medicine_info_task = Task(
    description="Retrieve structured details about the medicine: {medicine_name}.",
    expected_output="Structured JSON according to MedicineInfoOutput schema.",
    agent=rag_agent,
    output_pydantic=MedicineInfoOutput,
)

medicine_comparison_task = Task(
    description="Compare two medicines: {original_medicine} vs {alternative_medicine}.",
    expected_output="Structured JSON according to MedicineComparisonOutput schema.",
    agent=rag_agent,
    output_pydantic=MedicineComparisonOutput,
)

rag_chat_task = Task(
    description="Answer free-form user queries about medicines/diseases using RAG pipeline.",
    expected_output="A helpful, safe, and medically-informed response.",
    agent=rag_agent
)

# -------------------------
# Crew Setup
# -------------------------
medicine_rag_crew = Crew(
    agents=[rag_agent],
    tasks=[medicine_info_task, medicine_comparison_task, rag_chat_task],
    verbose=True
)

# -------------------------
# Helper Functions
# -------------------------
def answer_query(query: str):
    """Retrieve relevant docs + generate free-form answer."""
    docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a helpful medical assistant.
    Use the following retrieved medical knowledge to answer:

    Context:
    {context}

    Question: {query}
    Answer:"""

    return rag_agent.llm.invoke(prompt)

def get_medicine_info(medicine_name: str):
    """Run structured info task."""
    return medicine_info_task.run(medicine_name=medicine_name)

def compare_medicines(original_medicine: str, alternative_medicine: str):
    """Run structured comparison task."""
    return medicine_comparison_task.run(
        original_medicine=original_medicine,
        alternative_medicine=alternative_medicine
    )
