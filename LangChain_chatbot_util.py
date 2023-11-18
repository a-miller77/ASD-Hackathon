import os
import requests

from langchain.embeddings import HuggingFaceEmbeddings #for using HugginFace models
# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS  #facebook vectorizationfrom langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
#from langchain.vectorstores import Qdrant

import json
import textwrap
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

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
    
os.environ["HUGGINGFACEHUB_API_TOKEN"] = read_api_key()

class LLMFactory():
    _llm_models = {}
    _llm_tokenizers = {}
    _faq_chain = None
    _memory = None
    _context_db = None

    @staticmethod
    def initiate_model(model_name: str = "meta-llama/Llama-2-7b-chat-hf", api_key = None):
        if api_key != None:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        LLMFactory.get_model(model_name)

    @staticmethod
    def get_model(model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        if LLMFactory._llm_models.get(model_name, None) == None:

            print('importing model and tokenizer from HuggingFace')

            model, tokenizer = LLMFactory.__load_llm(model_name)

            LLMFactory._llm_models[model_name] = model
            LLMFactory._llm_tokenizers[model_name] = tokenizer

        return LLMFactory._llm_models.get(model_name)
    
    @staticmethod
    def get_tokenizer(model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        if LLMFactory._llm_tokenizers.get(model_name, None) == None:

            print('importing model and tokenizer from HuggingFace')

            model, tokenizer = LLMFactory.__load_llm(model_name)

            LLMFactory._llm_models[model_name] = model
            LLMFactory._llm_tokenizers[model_name] = tokenizer

        return LLMFactory._llm_tokenizers.get(model_name)
    
    @staticmethod
    def get_context_db(file_directory: str = './contexts', file_extension = '.txt'):
        if LLMFactory._context_db == None:
            LLMFactory._context_db = create_db_dir(file_directory, file_extension)
        return LLMFactory._context_db
    
    @staticmethod
    def get_pipeline(model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        return LLMFactory.__create_pipeline(model_name)
    
    @staticmethod
    def get_faq_chain(model_name: str = "meta-llama/Llama-2-7b-chat-hf", 
                      context = '',
                      create_new = False, 
                      memory=None):
        if create_new or LLMFactory._faq_chain == None:
            LLMFactory._faq_chain, LLMFactory._memory = \
                LLMFactory.__create_faq_chain(model_name, context, memory)
        return LLMFactory._faq_chain
    
    @staticmethod
    def get_current_chain_memory():
        if LLMFactory._faq_chain == None:
            LLMFactory.get_faq_chain()
        return LLMFactory._memory

    @staticmethod
    def __load_llm(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            use_auth_token=True,)
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map='auto',
                                                torch_dtype=torch.float16,
                                                use_auth_token=True,
                                                # load_in_8bit=True,
                                                # load_in_4bit=True
                                                )
        return model, tokenizer

    @staticmethod
    def __create_pipeline(llm_name, max_new_tokens=512, top_k=30, temperature=0):
        model = LLMFactory.get_model(llm_name)
        tokenizer = LLMFactory.get_tokenizer(llm_name)

        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer= tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        max_new_tokens = max_new_tokens,
                        do_sample=True,
                        top_k=top_k,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id
                    )

        return HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':temperature})
    
    def __create_faq_chain(model_name = "meta-llama/Llama-2-7b-chat-hf", context='', memory=None):
        instruction = """\n\nChat History:\n\n{chat_history} \n\nEnd Chat History\n\nUser: {user_input}"""
        system_prompt = """You are a helpful assistant, you always only answer for the 
        assistant then you stop. Read the chat history and reference the context provided 
        to better be able to respond to the prompt"""

        template = create_prompt(instruction, context, system_prompt)
        prompt = PromptTemplate(
            input_variables=["chat_history", "user_input"], template=template)
        if memory == None:
            memory = ConversationBufferMemory(memory_key="chat_history")
        return LLMChain(llm=LLMFactory.get_pipeline(model_name), 
                        prompt=prompt, verbose=True, memory=memory), memory

def get_context_docs(query: str, db):
    return db.similarity_search(query)

def create_db_dir(file_directory: str, file_extension = '.txt'):
    loader = DirectoryLoader(file_directory, glob=f"./*{file_extension}")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    docs = loader.load()
    docs_split = text_splitter.split_documents(docs)

    return FAISS.from_documents(docs_split, HuggingFaceEmbeddings())

def create_single_doc(file_name: str):
    loader = TextLoader(file_name)
    doc = loader.load()
    return doc

def create_prompt(instruction, context = '', new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    # print(B_INST, SYSTEM_PROMPT, context, instruction, E_INST)
    prompt_template =  B_INST + SYSTEM_PROMPT + context + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")

def generate(text, model, tokenizer):
    prompt = create_prompt(text)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs#, outputs

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        return wrapped_text +'\n\n'
        # return assistant_text







def llm_response(prompt: str):
    request_type = determine_request_type(prompt)
    if request_type == 'services': return 'begin provider programatical'
    if request_type == 'faq': return answer_faq(prompt)
    return 'I cannot help with that as it is outside the bounds of my expertise'



def determine_request_type(prompt: str) -> str:
    if is_service(prompt): return 'services'
    if is_faq(prompt): return 'faq'
    return 'services'

def is_service(prompt):
    doc = create_single_doc("./contexts/ASD_services")
    _, simularity = get_similarity(prompt, doc=doc)
    simularity = simularity[0]
    print(simularity)
    return simularity < 0.8

def is_faq(prompt: str) -> bool:
    db = create_db_dir("./contexts")
    _, simularity = get_similarity(prompt, db=db)
    return min(simularity) < 1.6

def answer_faq(user_prompt: str):
    #system_prompt = ''
    context_db = LLMFactory.get_context_db()
    context_list_db = get_context_docs(user_prompt, context_db)
    context = get_context_from_db(context_list_db)
    llm_chain = LLMFactory.get_faq_chain(create_new=True,
                                         context=context,
                                         memory=LLMFactory.get_current_chain_memory())
    return llm_chain.predict(user_input=user_prompt)

def get_context_from_db(db):
    header = f'\nContext:'
    footer = f'End Context\n\n'
    for doc in db:
        header = f'{header}\n\n {doc.page_content}'
    return header + footer

def get_similarity(query, doc=None, db=None):
    """
    Returns a list of simularities and all documents given by either the doc or db paramter. Only one
    of the variables can be used, not both.
    """
    assert (doc==None or db==None) and not (doc==None and db==None)
    if db == None:
        db = FAISS.from_documents(doc, HuggingFaceEmbeddings())
    db_docs = db.similarity_search_with_score(query)
    simularity_list = []
    doc_list = []
    
    for item in db_docs:
        simularity, doc = item
        simularity_list.append(simularity)
        doc_list.append(doc)
    return simularity_list, doc_list