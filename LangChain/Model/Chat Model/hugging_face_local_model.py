from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache' # download and store in this path

llm = HuggingFacePipeline.from_model_id(
        model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', # model name
        task = 'text-generation',
        pipeline_kwargs = dict(temperature = 0.5,
                               max_new_tokens = 100)
        )

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)
