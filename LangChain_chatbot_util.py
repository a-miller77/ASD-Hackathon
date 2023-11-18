import os
import requests

from langchain.embeddings import HuggingFaceEmbeddings #for using HugginFace models
# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS  #facebook vectorizationfrom langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
#from langchain.vectorstores import Qdrant

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from LangChain_chatbot_util import read_api_key

os.environ["HUGGINGFACEHUB_API_TOKEN"] = read_api_key()

def return_context_docs(query: str, db):
    return db.similarity_search(query)

def create_db(file_directory: str):
    loader = DirectoryLoader('./contexts', glob="./*.txt")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    docs = loader.load()
    docs_split = text_splitter.split_documents(docs)

    return FAISS.from_documents(docs_split, HuggingFaceEmbeddings())

def create_faq_chain():
    instruction = "Chat History:\n\n{chat_history} \n\nEnd Chat History\n\nUser: {user_input}"
    system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. Read the chat history to get context"

    template = get_prompt(instruction, system_prompt)
    print(template)

def read_api_key(file_path='api-key.txt'):
    try:
        with open(file_path, 'r') as file:
            api_key = file.read().strip()
            return api_key
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None