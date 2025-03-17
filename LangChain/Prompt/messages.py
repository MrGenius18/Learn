from langchain_core.messages import *
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

messages = [
    SystemMessage(content='You are a helpful assistant')
]

user_ip = input("enter your Query: ")
messages.append(HumanMessage(content=user_ip))

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(result.content)