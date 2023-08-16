# Question Answering Bot 
Question Answering bot is implemented with LangChain and OpenAI. The bot takes in documents and questions, and outputs answers to those questions according to uploaded documents. It also supports summarization of documents, and translation of answers. 

## Hallucination 
The bot reduces hallucination by comparing two answers given by LangChain and OpenAI API. If two answers have the same meaning, the bot will provide an answer to questions. Otherwise, the bot will give both of answers to users and give users a warning about the accuracy of answers. 

## Run web demo 
1. Clone this repository.  
   
   ```git clone https://github.com/panyang05/Question-Answering-System.git```

2. Install required dependencies. 
   
   ```pip install -r requirements.txt```

3. Go to ```website``` directory 
   
   ```cd website```

4. Run Flask server
   
   ```python main.py```
   
5. Go to the address shown in terminal. For example, http://127.0.0.1:12345
