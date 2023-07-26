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
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredMarkdownLoader


template = """Answer the question based on the context below. You are NOT allowed to use any outside information. If the question cannot be answered using the information provided, you must answer with "I don't know".

Context: {context}

Question: {query}

Answer: """

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)



def long_question_answer(openai_key, questions):
    try:
        os.environ["OPENAI_API_KEY"] = openai_key
        embeddings = OpenAIEmbeddings()
        bard = Bard(token="YwhCST9bVl4ap4RL5_gQ-GTotXrYhf7_04CpVx2IlyFyr2b2dWXoa9GEems1Vhor1VHjdA.")
        evaluator = Bard(token="YwhCST9bVl4ap4RL5_gQ-GTotXrYhf7_04CpVx2IlyFyr2b2dWXoa9GEems1Vhor1VHjdA.")
        text = ""
        pages = []
        temp = []
        for question in questions.split("\n"):
            for filename in os.listdir('static/upload'):
                extension = filename.split('.')[-1]
                if extension == 'md':
                    loader = UnstructuredMarkdownLoader('static/upload/'+filename)
                    data = loader.load()
                    db = Chroma.from_documents(data, embeddings)
                    docs = db.similarity_search(question)
                    pages = pages + [each.page_content for each in docs]

                elif extension == 'pdf':
                    loader = PyPDFLoader('static/upload/'+filename)
                    data = loader.load_and_split()
                    db = Chroma.from_documents(data, embeddings)
                    docs = db.similarity_search(question)
                    pages = pages + [each.page_content for each in docs]
                elif extension == 'html':
                    loader = UnstructuredHTMLLoader('static/upload/'+filename)
                    data = loader.load()
                    db = Chroma.from_documents(data, embeddings)
                    docs = db.similarity_search(question)
                    pages = pages + [each.page_content for each in docs]
                    
                elif extension == 'csv':
                    loader = CSVLoader(file_path='static/upload/'+filename)
                    data = loader.load_and_split()
                    db = Chroma.from_documents(data, embeddings)
                    docs = db.similarity_search(question)
                    pages = pages + [each.page_content for each in docs]
                else:
                    with open('static/upload/'+filename) as f:
                        text += f.read()
                    text += "=====================\n\n"
                    text = CharacterTextSplitter().split_text(text)
                    data =  [Document(page_content=t) for t in text]
                    db = Chroma.from_documents(data, embeddings)
                    docs = db.similarity_search(question)
                    pages = pages + [each.page_content for each in docs]

            
            model = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k",)
            
        
            answer1 = ""
            answer2 = ""
            for page in docs:
                text = page.page_content
                in_text = prompt_template.format(context=text, query=question)
                res_text = model(in_text)
                if  "I don't know".lower() not in res_text.lower():
                    answer1 += res_text + ' '

                res_text = bard.get_answer(in_text)['content']

                if "I don't know".lower() not in res_text.lower():
                    answer2 += res_text + ' '
            if len(answer1) == 0 or len(answer2) == 0:
                answer = ["I don't know."]
            else:
                eval = evaluator.get_answer(f'Yes or No: "{answer1}" and {answer2} have the same meaning)')['content']
                if "yes" in eval.lower():
                    answer = [model(f'Summarize the following text: {answer1 + " " + answer2}')]
                else:
                    answer = [f"Both answers are possible, please check carefully:",
                            f"answer1: {answer1}",
                            f"answer2: {answer2}"]
            
            temp.append((question, answer))
        return temp
    except:
        return "Something went wrong. Please try again!"


def summarization(openai_key, filename):
    try:
        os.environ["OPENAI_API_KEY"] = openai_key

        text = ""

        extension = filename.split('.')[-1]
        if extension == 'md':
            loader = UnstructuredMarkdownLoader('static/upload/'+filename)
            pages = loader.load_and_split()


        elif extension == 'pdf':
            loader = PyPDFLoader('static/upload/'+filename)
            pages = loader.load_and_split()
        elif extension == 'html':
            loader = UnstructuredHTMLLoader('static/upload/'+filename)
            pages = loader.load_and_split()
            
        elif extension == 'csv':
            loader = CSVLoader(file_path='static/upload/'+filename)
            pages = loader.load_and_split()
        else:
            with open('static/upload/'+filename) as f:
                text += f.read()
            text += "=====================\n\n"
            text = CharacterTextSplitter().split_text(text)
            pages = [Document(page_content=t) for t in text]
        chain = load_summarize_chain(ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0), chain_type="map_reduce")
        res_text = chain.run(pages)

        return [[res_text]]
    except:
        return [["Something went wrong. Please try again!"]]


def translation(openai_key, outlanguage, res_texts):
    try:
        os.environ["OPENAI_API_KEY"] = openai_key
        result = []
        for i in range(len(res_texts)):
            temp = []
            for j in range(len(res_texts[i])):
                res_text = res_texts[i][j]
                translated = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": f"You are a {outlanguage} translator."},
                        {"role": "user", "content": f"I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in {outlanguage}. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level f{outlanguage} words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations. The paragrah you will translate is {res_text}."}
                    ]
                )
                temp.append(translated["choices"][0]["message"]["content"])
            result.append(temp.copy())

        return result
    except:
        return [["Something went wrong. Please try again!"]]
    


def translation_qa(openai_key, outlanguage, res_texts):
    try:
        os.environ["OPENAI_API_KEY"] = openai_key
        result = []
        for i in range(len(res_texts)):
            temp = []
            for j in range(len(res_texts[i][1])):
                res_text = res_texts[i][1][j]
                translated = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": f"You are a {outlanguage} translator."},
                        {"role": "user", "content": f"I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in {outlanguage}. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level f{outlanguage} words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations. The paragrah you will translate is {res_text}."}
                    ]
                )
                temp.append(translated["choices"][0]["message"]["content"])
            result.append((res_texts[i][0], temp.copy()))

        return result
    except:
        return [["Something went wrong. Please try again!"]]