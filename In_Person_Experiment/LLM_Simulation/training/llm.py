"""Helpers used by the Q50 training runner."""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_openai(model, base_url, api_key):

    llm = ChatOpenAI(model = model, base_url=base_url, api_key=api_key)
    return llm

def create_decision_assistant(model):
    system_prompt = (
    "You will be given responses from a chatbot that follow a specific format. Each response may contain the reasoning process and the final answer."
    "The final decision will either be ORDER or NOT ORDER. You SHOULD ONLY extract the final answer."
    "If the final decision is ORDER, return '1' with no additional commentary in your response."
    "If the final decision is NOT ORDER, return '0' with no additional commentary in your response."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    chain = prompt | model | StrOutputParser()
    return chain
