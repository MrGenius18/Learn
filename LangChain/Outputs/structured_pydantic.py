from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, EmailStr, Field

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# schema ############################
class Review(BaseModel): # (TypedDict)

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "Abrief summary of the review"] # gives extra prompt to guide model
    sentiment: Literal["Positive", "Negative", "Neutral"]  # provide options
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Optional[list[str]] = Field(description="Write down all the cons inside a list")
    sentiment_score: int = Field(gt=0, lt=5, description="based on  reveiew sentiment")
    Reviewer: str = 'Bhautik' # set by default value
    email: Optional[EmailStr] = Field(default=None) # if mail-id provided

structure_model = model.with_structured_output(Review)
######################################################

str = """
        I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The SØØØmAh battery easily lasts a full day even with heavy use, and the 4SW fast charging is a lifesaver."

        The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 2ØØt•1P camera—the night mode is stunning, capturing crisp, vibrant images even in Iow light. Zooming up to Iøøx actually works well for distant objects, but anything beyond 30x loses quality.

        However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,30 price tag is also a hard pill to swallow.

        pros :
        Insanely powerful processor (great for gaming and productivity)
        Stunning 2ØØMP camera with incredible zoom capabilities
        Long battery life with fast charging
        S-Pen support is unique and useful
    """

result = structure_model.invoke(str)
result_json = result.model_dump_json()

print(result)
print("\n", result.sentiment, "\n")
