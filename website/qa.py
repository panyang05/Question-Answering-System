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


def long_question_answer(openai_key, question):
    os.environ["OPENAI_API_KEY"] = openai_key

    text = ""

    for filename in os.listdir('static/upload'):
        extension = filename.split('.')[-1]
        if extension == 'txt':
            with open('static/upload/'+filename) as f:
                text += f.read()
            text += "=====================\n\n"
            text = CharacterTextSplitter().split_text(text)
            pages = [Document(page_content=t) for t in text]


        elif extension == 'pdf':
            loader = PyPDFLoader('static/upload/'+filename)
            pages = loader.load_and_split()

    model = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k",)
    answer = ""
    for page in pages:
        text = page.page_content
        in_text = prompt_template.format(context=text, query=question)
        res_text = model(in_text)
        if res_text != "I don't know.":
            answer += res_text + ' '
    if len(answer) == 0:
        answer = "I don't know."
    else:
        answer = model(f'Summarize the following text: {answer}')

    return answer


def summarization(openai_key, filename):
    os.environ["OPENAI_API_KEY"] = openai_key

    text = ""

    extension = filename.split('.')[-1]
    if extension == 'txt':
        with open('static/upload/'+filename) as f:
            text += f.read()
        text += "=====================\n\n"
        text = CharacterTextSplitter().split_text(text)
        pages = [Document(page_content=t) for t in text]
    elif extension == 'pdf':
        loader = PyPDFLoader('static/upload/'+filename)
        pages = loader.load_and_split()

    chain = load_summarize_chain(ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0), chain_type="map_reduce")
    res_text = chain.run(pages)

    return res_text


def translation(openai_key, outlanguage, res_text):
    os.environ["OPENAI_API_KEY"] = openai_key
    translated = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": f"You are a {outlanguage} translator."},
            {"role": "user", "content": f"I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in {outlanguage}. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level f{outlanguage} words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations. The paragrah you will translate is {res_text}."}
        ]
    )

    return translated["choices"][0]["message"]["content"]