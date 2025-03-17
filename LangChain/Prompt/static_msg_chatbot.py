from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import *
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

chat_history = [
    SystemMessage(content=f"You are a Assistant")
]

while True:
    user_ip = input('Ask Me (if not a query enter "exit"): ')
    if user_ip == 'exit':
        break
    chat_history.append(HumanMessage(content=user_ip))

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print(chat_history)