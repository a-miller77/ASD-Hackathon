import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain

from LangChain_chatbot_util import *

import os
import textwrap
from clinic_match import ClinicMatch

class ClinicMatch:
    def __init__(self, token: str, file_path = './big_data_energy/provider_info.csv', source = None) -> None:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        embeddings = HuggingFaceEmbeddings()
        del os.environ["HUGGINGFACEHUB_API_TOKEN"]

        data = CSVLoader(
            file_path=file_path, 
            source_column=source
        ).load()

        self.db = FAISS.from_documents(data, embeddings)

    
    def query(self, query: str) -> 'list[Document]':
        return self.db.similarity_search(query)
    

class ClinicQuery:
    def __init__(self, token: str, file_path = './big_data_energy/provider_info.csv', source = None) -> None:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        embeddings = HuggingFaceEmbeddings()
        del os.environ["HUGGINGFACEHUB_API_TOKEN"]

        tokenizer = AutoTokenizer.from_pretrained("t5-small",
                                          use_auth_token=True,)
        
        pip = pipeline("text-generation", model="lmsys/vicuna-7b-v1.3", tokenizer=tokenizer, device_map="auto",
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
        

    def query_providers(text):

        #### Single prompt only
        system_prompt = """\
            You are an internal system key search terms assistant who is searching for keywords to search a table for.
            You categorize information in questions into categories following system format. If nothing is found for a category, please leave the section blank. Do not fill blank categories. Additional Categories do not need to be added if blank. 
            PLEASE do not add additional keywords that are not explicitly stated in the entry.

            System Format:

            Required Categories: -age: (Toddler, Child, Adolescent, Adult), -service: (Therapy, Assessment, Diagnosis, Consultation, Advocacy, Educational)
            Additional Categories: -s: (service specific), -insurance: ( ), -language: ( ), -a: (additional keywords)
            
            End System Format

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            """

        instruction = "Please catagorize the keywords in the request. Please start and end the list with START and END tags: \n\n {text}"
        template = get_prompt(instruction, system_prompt)
        # print(template)

        prompt = PromptTemplate(template=template, input_variables=["text"])

        llm_chain = LLMChain(prompt=prompt, llm=HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0}))

        output = llm_chain.run(text)
        # output = generate(text, template)

        return parse_text(output)