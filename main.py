import os
import fitz  # PyMuPDF
import pytesseract
#from pdfplumber import pdf
import pdfplumber  # ✅ Correct import

from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import json
from multiprocessing import Pool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

app = FastAPI()

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

# OCR for images
def extract_text_from_image(image_path):
    return pytesseract.image_to_string(image_path)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract tables from PDF
def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:  # ✅ Corrected
        for page in pdf.pages:
            tables.extend(page.extract_tables())
    return tables

# Process a single document
def process_document(file_path: str):
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
        tables = extract_tables_from_pdf(file_path)
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        text = extract_text_from_image(file_path)
        tables = []
    else:
        with open(file_path, "r") as f:
            text = f.read()
        tables = []
    return {"text": text, "tables": tables}

# Batch processing
def process_documents(file_paths: List[str]):
    results = [process_document(file) for file in file_paths]  # ✅ Safer approach
    return results

'''def process_documents(file_paths: List[str]):
    with Pool() as pool:
        results = pool.map(process_document, file_paths)
    return results'''

# API endpoint for document upload
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    file_paths = []
    for file in files:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(file_path)
    processed_data = process_documents(file_paths)
    with open("processed_data.json", "w") as f:
        json.dump(processed_data, f)
    return {"message": "Documents processed successfully", "data": processed_data}

# Load processed documents or initialize an empty list
def load_processed_data():
    if os.path.exists("processed_data.json"):
        with open("processed_data.json", "r") as f:
            return json.load(f)
    return []

# Initialize QA system

def initialize_qa_system():
    documents = load_processed_data()
    if not documents:
        raise HTTPException(status_code=400, detail="No documents processed yet. Please upload documents first.")
    # Check if there is text data
    if not any(doc["text"] for doc in documents):
        raise HTTPException(status_code=400, detail="Processed documents contain no text.")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    texts = [doc["text"] for doc in documents]
    chunks = text_splitter.split_text("\n".join(texts))

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 results for better context

    # Initialize LLM
    #llm = HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation")
    '''llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    model_kwargs={
        "max_length": 1024,  # Allow up to 1024 tokens (increase as needed)
        "temperature": 0.7,  # Adjust creativity
        "top_k": 50,  # Use top-k sampling
        "top_p": 0.95  # Use nucleus sampling
    }
)'''
   
    from transformers import pipeline

    llm_pipeline = pipeline(
      "text-generation",
      model="gpt2",
      max_length=1024,  # Allow longer input
      temperature=0.6,  # Adjust creativity
      top_k=40,
      top_p=0.9
)

    llm = HuggingFacePipeline(pipeline=llm_pipeline)


    # Create QA system with memory
    #memory = ConversationBufferMemory()
    #memory = ConversationBufferMemory(return_messages=True)  # ✅ Fix deprecation
    memory = ConversationBufferMemory(return_messages=True, output_key="result")  # ✅ Fix
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        #retriever=vector_store.as_retriever(),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="result"  # ✅ Fix
    )
    return qa

# Answer questions
def answer_question(question: str):
    qa = initialize_qa_system()
    result = qa({"query": question})
    return {
        "answer": result["result"],
        "source": result["source_documents"][0].metadata if result["source_documents"] else None
    }

# API endpoint for question answering
'''@app.post("/ask/")
async def ask_question(question: str):
    try:
        response = answer_question(question)
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''
    
@app.post("/ask/")
async def ask_question(question: str):
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        response = answer_question(question)
        return response

    except HTTPException as e:
        raise e  # Let FastAPI handle known errors

    except Exception as e:
        print(f"Error in /ask/: {e}")  # Debugging
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)