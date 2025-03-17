from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain}'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])
chain = chat_template | model

# load chat history
chat_history = []
with open('chat_history.txt', 'a+') as f: # 'a+' mode creates the file if it doesn’t exist.
    f.seek(0) # Move cursor to the beginning to read
    chat_history.extend(f.readlines())

user_query = input("Enter Your Query: ")

# create prompt
result = chain.invoke({'domain':'customer support agent',
                       'chat_history': chat_history,
                       'query': user_query})
print(result.content)
