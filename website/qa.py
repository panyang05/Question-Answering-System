from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
import os
from bardapi import Bard
from langchain.document_loaders import PyPDFLoader
import openai
from langchain.llms import OpenAI

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

template = """Answer the question based on the context below. You are NOT allowed to use any outside information. If the question cannot be answered using the information provided, you must answer with "I don't know".

Context: {context}

Question: {query}

Answer: """

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)

def question_answer(openai_key, question):
    os.environ["OPENAI_API_KEY"] = openai_key

    text = ""

    for filename in os.listdir('static/upload'):
        extension = filename.split('.')[-1]
        if extension == 'txt':
            with open('static/upload/'+filename) as f:
                text += f.read()
            text += "=====================\n\n"
        elif extension == 'pdf':
            loader = PyPDFLoader('static/upload/'+filename)
            pages = loader.load_and_split()

            for each in pages:
                text += each.page_content
            text += "=====================\n\n"

    model = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k",)
    in_text = prompt_template.format(context=text, query=question)
    res_text = model(in_text)

    return res_text

def summerization(openai_key):
    os.environ["OPENAI_API_KEY"] = openai_key

    text = ""

    for filename in os.listdir('static/upload'):
        extension = filename.split('.')[-1]
        if extension == 'txt':
            with open('static/upload/'+filename) as f:
                text += f.read()
            text += "=====================\n\n"
        elif extension == 'pdf':
            loader = PyPDFLoader('static/upload/'+filename)
            pages = loader.load_and_split()

            # for each in pages:
            #     text += each.page_content
            # text += "=====================\n\n"

    

    chain = load_summarize_chain(ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0), chain_type="map_reduce")
    res_text = chain.run(pages)

    return res_text
    