import docx
from langchain.text_splitter import NLTKTextSplitter
import pandas as pd
from llama_index import download_loader

text_splitter = NLTKTextSplitter(chunk_size=1000)

def get_document_text(filename):
    if filename.endswith('.docx'):
        doc = docx.Document(filename)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text_splitter.split_text(text)
    if filename.endswith('.txt'):
        with open(filename, 'r') as f:
            return text_splitter.split_text(f.read())
    if filename.endswith('.csv'):
        return get_csv_data(filename)
    return None

def split_text(text):
    return text_splitter.split_text(text)

def get_csv_data(filename):
    return pd.read_csv(filename, engine="python", encoding="ISO-8859-1").to_string().strip()