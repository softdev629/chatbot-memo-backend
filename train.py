import dotenv
import pandas as pd
from pandas import DataFrame
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

load_dotenv()

vector_db = None

if os.path.exists("./store/index.faiss"):
    vector_db = FAISS.load_local("./store", OpenAIEmbeddings())
else:
    vector_db = FAISS.from_documents([Document(page_content="This is ZK-Rollup Crypto Info Data.\n\n")], OpenAIEmbeddings())

extra_qa_info: DataFrame = pd.read_excel("./traindata/qa.xlsx")
qa_list = extra_qa_info.to_dict(orient="records")
for qa_item in qa_list:
    document = Document(page_content=f"Q: {qa_item['Question']}\nA: {qa_item['Answer']}")
    vector_db.add_documents([document])

extra_crypto_info: DataFrame = pd.read_excel("./traindata/given.xlsx")
info_list = extra_crypto_info.to_dict(orient="records")
text = ""
for info_item in info_list:
    symbol = info_item['Token Symbol']
    text += f"This is ZK-Rollup named {info_item['Name']}. Token Symbol of {info_item['Name']} is {symbol}. {symbol} was last taken on {info_item['Date'].date()}. {symbol} has {'' if info_item['Released'] else 'not '}been released. Price of {symbol} is ${info_item['Price']}. Price change of {symbol} in last week is {info_item['7d Change'] * 100}%. Price change of {symbol} in last month is {info_item['30d Change'] * 100}%. Total market value (Market Cap) of {symbol}'s circulating supply is ${info_item['Market Cap']}. Trading amount(Volume) traded in the last 24 hours is ${info_item['Volume']}. Amount of coins circulating in market and public hands of {symbol} is {info_item['Supply']}. {info_item['Description']}\n\n"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_text(text)
for doc_item in docs:
    document = Document(page_content=doc_item)
    vector_db.add_documents([document])

vector_db.save_local("./store")
print("completed")