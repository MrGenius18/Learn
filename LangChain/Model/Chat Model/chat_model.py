from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

# model = ChatOpenAI(model='gpt-4')
# model = ChatAnthropic(model='claude-3-5-sonnet-20241022')
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', 
                               temperature=0.6, # temperature is controls the randomness o/p. range(0-1.5)
                               max_completion_tokens=10) # maximmum word size

result = model.invoke("What is the capital of India")

print(result.content)