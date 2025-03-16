# from langchain_openai import OpenAI
from langchain_google_genai  import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# llm = OpenAI(model='gpt-3.5-turbo-instruct')
llm = GoogleGenerativeAI(model='gemini-2.0-flash')

result = llm.invoke("What is the capital of India")

print(result)