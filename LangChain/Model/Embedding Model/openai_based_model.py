from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', 
                             dimensions=200) # vector size

text = "Delhi is the capital of India"

vector = embedding.embed_query(text)
print(str(vector))

# multiple queries embending
documents = ["Delhi is the capital of India",
             "Kolkata is the capital of West Bengal",
             "Paris is the capital of France"]

vectors = embedding.embed_documents(documents)
print(str(vectors))