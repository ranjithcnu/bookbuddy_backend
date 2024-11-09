from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import re
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List

app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin. Replace "*" with specific origins for better security.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods.
    allow_headers=["*"],  # Allows all headers.
)

model = SentenceTransformer('all-MiniLM-L6-v2')
chunks = []
embeddings = None

# Utility functions
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def preprocess_text(text):
    """Preprocesses the extracted text by removing multiple spaces and newlines."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_length=500):
    """Splits the text into chunks of a specified maximum length."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(' '.join(current_chunk + [sentence])) <= max_length:
            current_chunk.append(sentence)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def build_vector_store(chunks, model):
    """Builds a vector store from text chunks using embeddings."""
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return chunks, embeddings

def answer_question(question, chunks, embeddings, model, top_k=3):
    """Finds the most relevant chunks to the question and returns a response."""
    question_embedding = model.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]

    relevant_chunks = [chunks[idx] for idx in top_results]
    response = ' '.join(relevant_chunks)
    return response

# FastAPI endpoints
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):
    global chunks, embeddings
    if file.content_type not in ["application/pdf", "application/octet-stream"] or \
       (file.content_type == "application/octet-stream" and not file.filename.endswith(".pdf")):
        return JSONResponse(content={"error": "Invalid file type. Only PDFs are accepted."}, status_code=400)

    # Save the PDF file and extract its text
    file_location = f"uploaded_files/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Extract and process text from the PDF
    raw_text = extract_text_from_pdf(file_location)
    clean_text = preprocess_text(raw_text)
    text_chunks = chunk_text(clean_text)
    
    # Build the vector store with embeddings
    chunks, embeddings = build_vector_store(text_chunks, model)
    
    return {"message": "File uploaded and text processed successfully"}

@app.post("/answer")
async def get_answer(request: Request):
    global chunks, embeddings
    try:
        data = await request.json()
        question = data.get("question")
        if not question:
            raise HTTPException(status_code=422, detail="Missing 'question' field")
        
        if not chunks or embeddings is None:
            raise HTTPException(status_code=500, detail="No PDF text found. Please upload a PDF first.")
        
        # Generate the answer
        response = answer_question(question, chunks, embeddings, model)
        return {"answer": response}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
