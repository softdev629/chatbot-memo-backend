import os
from flask import Flask, request, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post('/api/upload')
def upload():
    if 'file' not in request.files:
        return {"state": "error", "message": "No file part"}
    file = request.files['file']
    if file.filename == '':
        return {"state": "error", "message": "No selected file"}
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return {"state": "success"}
    return {"state": "error", "message": "Invalid file format"}

