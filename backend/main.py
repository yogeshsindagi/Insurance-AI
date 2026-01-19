from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import joblib
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai.chat_models import ChatMistralAI

from dotenv import load_dotenv

load_dotenv()  # this loads .env into environment

app = FastAPI(title="Insurance AI Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, GET, OPTIONS
    allow_headers=["*"],
)


# ------------------ RAG SETUP ------------------

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(
    "insurance_faiss_db",
    embeddings,
    allow_dangerous_deserialization=True   # you created this, so it's safe
)

llm = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0.1,
    api_key=os.getenv("MISTRAL_API_KEY")
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    docs = db.similarity_search(req.question, k=3)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are an expert Indian health insurance assistant.

Context:
{context}

Question:
{req.question}

Answer:
"""
    response = llm.invoke(prompt)
    return {"answer": response.content}

# ------------------ ML PREMIUM MODEL ------------------

model = joblib.load("insurance_premium_model.pkl")

class PredictRequest(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    disease: int
    policy_type: int

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    premium = model.predict(df)[0]
    return {"premium": int(premium)}

@app.get("/")
def root():
    return {"status": "Insurance RAG + Premium Prediction API Running"}
