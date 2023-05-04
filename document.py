import docx
from langchain.text_splitter import NLTKTextSplitter
    
text_splitter = NLTKTextSplitter(chunk_size=1000)

def get_document_text(filename):
    if filename.endswith('.docx'):
        doc = docx.Document(filename)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    if filename.endswith('.txt'):
        with open(filename, 'r') as f:
            return f.read()
    return None

def split_text(text):
    return text_splitter.split_text(text)