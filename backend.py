from fastapi import FastAPI
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from pydantic import BaseModel
import os

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


DB_FAISS_PATH = 'vectorstore/db_faiss'
CSV_FILE_PATH = 'data/q2sex.csv' 

class QueryRequest(BaseModel):
    question: str
    chat_history: list = []

# Load the LLM
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Loading the csv file
@app.on_event("startup")
async def load_csv_on_startup():
    if os.path.exists(CSV_FILE_PATH):
        loader = CSVLoader(file_path=CSV_FILE_PATH, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)
        print(f"CSV loaded and FAISS index created from {CSV_FILE_PATH}.")
    else:
        print(f"CSV file not found at {CSV_FILE_PATH}")

# API endpoint
@app.post("/ask/")
async def ask_question(query_request: QueryRequest):
    db = FAISS.load_local(DB_FAISS_PATH, HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'}),allow_dangerous_deserialization=True)

    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
    
    result = chain.invoke({"question": query_request.question, "chat_history": query_request.chat_history})

    return {
        "answer": result["answer"],
        "chat_history": query_request.chat_history + [(query_request.question, result["answer"])]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
