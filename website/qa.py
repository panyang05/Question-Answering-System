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
from langchain.document_loaders import PyPDFLoader
import openai
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from tqdm import tqdm
from langchain.document_loaders import BSHTMLLoader


template = """Answer the question based on the context below. You are NOT allowed to use any outside information. If the question cannot be answered using the information provided, you must answer with "I don't know".
Context: {context}
Question: {query}
Answer: """
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)


def long_question_answer(openai_key, questions):
    # try:
        os.environ["OPENAI_API_KEY"] = openai_key

        temp = []
        
        pages = []
        text = ""
        all_data = []
        for filename in tqdm(os.listdir('static/upload')):
            extension = filename.split('.')[-1]
            if extension == 'md':  
                loader = UnstructuredMarkdownLoader('static/upload/'+filename)
                data = loader.load()
                pages = pages + [each.page_content for each in data]
            elif extension == 'pdf':
                loader = PyPDFLoader('static/upload/'+filename)
                data = loader.load_and_split()
                pages = pages + [each.page_content for each in data]
            elif extension == 'html':
                loader = BSHTMLLoader('static/upload/'+filename)
                data = loader.load_and_split()
                pages = pages + [each.page_content for each in data]
                
            elif extension == 'csv':
                loader = CSVLoader(file_path='static/upload/'+filename)
                data = loader.load_and_split()
                pages = pages + [each.page_content for each in data]
            else:
                with open('static/upload/'+filename) as f:
                    text += f.read()
                text += "=====================\n\n"
                text = CharacterTextSplitter().split_text(text)
                data =  [Document(page_content=t) for t in text]
                pages = pages + [each.page_content for each in data]
            all_data += data

        for question in tqdm(questions.split("\n")):
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            db = Chroma.from_documents(all_data, embeddings)
            docs = db.similarity_search(question)
            pages = [each.page_content for each in data]

            model = ChatOpenAI(model="gpt-4", temperature=0)
            chain = load_qa_with_sources_chain(llm=model, chain_type="map_reduce")
            langchain_res = chain({"input_documents": docs, "question": question}, return_only_outputs=True)['output_text']
            
            openai.api_key = openai_key
            prompt = f"""Answer the question briefly given the context below as {{Context:}}. \n
                If the answer is not available in the {{Context:}} and you are not confident about the output,
                please say "Information not available in provided context". \n\n
                Context: {pages}\n
                Question: {question} \n
                Answer:
                """
            chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
            prompt_res = chat_completion.choices[0].message.content

            eval_prompt = f"""Compare {{Statement1}} and {{Statement2}} below. If {{Answer1}} and {{Answer2}} have the same meaning, please say 'yes'. If they have different meaning, please say 'no'
                            Statement1: {langchain_res}\n
                            Statement2: {prompt_res}\n
                            Answer:
                            """
            eval_res = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": eval_prompt}])
            eval_res = eval_res.choices[0].message.content

            if "yes" in eval_res.lower():
                answer = [langchain_res]
            else:
                answer = [f"Both answers are possible, please check carefully:",
                             f"Answer1: {langchain_res}",
                             f"Answer2: {prompt_res}"]
            temp.append((question, answer))
        return temp

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
        chain = load_summarize_chain(ChatOpenAI(model_name="gpt-4", temperature=0), chain_type="map_reduce")
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
                model="gpt-4",
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