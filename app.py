import os
from flask import Flask, request, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

import pickle

UPLOAD_FOLDER = './pdfs'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload():
    global reader, raw_text, texts, embeddings, docsearch

    if 'file' not in request.files:
        return {"state": "error", "message": "No file part"}
    file = request.files['file']
    if file.filename == '':
        return {"state": "error", "message": "No selected file"}
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        reader = PdfReader(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.split_text(raw_text)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        return {"state": "success"}
    return {"state": "error", "message": "Invalid file format"}

@app.route('/api/chat', methods=['POST'])
def chat():
    query = request.form["prompt"]
    docs = docsearch.similarity_search(query)
    completion = chain.run(input_documents=docs, question=query)
    return {"answer": completion }