from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

def get_openai(model, base_url, api_key):

    llm = ChatOpenAI(model = model, base_url=base_url, api_key=api_key)
    return llm

def init_history(n_cohort, human_msg1, n=4):
    '''
    This function initiates conversations for (n_cohort*n) participants.
    The experiment simulation will take place in those conversations.
    '''
    store_dict = {}
    for i in range(n_cohort):
        for k in range(n):
            key = f'{i+1},{k+1}'
            store_dict[key] = InMemoryChatMessageHistory(
                messages = [HumanMessage(content = human_msg1)])
    return store_dict

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




