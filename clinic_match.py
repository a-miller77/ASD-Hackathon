import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

class ClinicMatch:
    def __init__(self, token: str) -> None:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        embeddings = HuggingFaceEmbeddings()
        del os.environ["HUGGINGFACEHUB_API_TOKEN"]

        data = CSVLoader(
            file_path='./big_data_energy/our_data2.csv', 
            source_column="Site"
        ).load()

        self.db = FAISS.from_documents(data, embeddings)

    
    def query(self, query: str) -> 'list[Document]':
        return self.db.similarity_search(query)