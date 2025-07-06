import os
import sys
import re
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LeadScore(BaseModel):
    score: int = Field(description="The lead score from 1 to 10.")
    reasoning: str = Field(description="The reasoning for the assigned score.")
    key_factors: List[str] = Field(description="The key factors from the document that influenced the score.")

def get_lead_scorer(api_key: str):
    """Initializes and returns the lead scoring chain using the OpenRouter API."""
    
    llm = ChatOpenAI(
        model="deepseek/deepseek-chat:free",
        api_key=api_key,  # type: ignore
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=1024,
    )

    parser = JsonOutputParser(pydantic_object=LeadScore)

    prompt_template = """
You are a highly experienced financial analyst specializing in lead qualification.
Your task is to analyze the provided text from a financial document and assign a lead score from 1 to 10, where 1 indicates a very low-quality lead and 10 indicates a very high-quality lead.

Base your score on the presence of specific, actionable financial details such as loan amounts, investment sizes, contract terms, company valuations, and clear financial health indicators. High-quality leads are documents with concrete, significant financial data. Low-quality leads are generic, lack specific numbers, or are not related to financial transactions.

Carefully analyze the document and provide a detailed reasoning for your score, referencing the specific factors you identified. You must provide your answer in the JSON format requested.

DOCUMENT_TEXT:
{document_text}

{format_instructions}
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["document_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    return chain

def score_lead(document_path: str, chain) -> Dict[str, Any]:
    """Scores a single lead document."""
    try:
        with open(document_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        document = Document(page_content=content, metadata={"source": document_path})
        
        result = chain.invoke({"document_text": document.page_content})
        result['source'] = os.path.basename(document_path)
        return result

    except FileNotFoundError:
        return {"error": "File not found.", "source": os.path.basename(document_path)}
    except Exception as e:
        return {"error": str(e), "source": os.path.basename(document_path)}

if __name__ == '__main__':
    # Example Usage
    try:
        # We'll take the content from the first mock document to test
        with open("mock_docs/document_1.txt", "r", encoding="utf-8") as f:
            sample_doc = f.read()
        
        chain = get_lead_scorer("your_api_key_here")
        print("--- Scoring sample document ---")
        result = score_lead("mock_docs/document_1.txt", chain)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Lead Score: {result['score']}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Key Factors: {', '.join(result['key_factors'])}")

    except (ValueError, FileNotFoundError) as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 