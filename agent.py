# -*- coding: utf-8 -*-
"""agents.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_5pW9aa7dmA0P-puKgbyEf-M2IY908E0
"""

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
tools = load_tools(["terminal"])

agent_chain = initialize_agent(
    tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

result = agent_chain.run("sample_data ディレクトリにあるファイルの一覧を教えて")
print(result)

# Multi Functions Agent
chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
tools = load_tools(["ddg-search"])
agent_chain = initialize_agent(
    tools, chat, agent=AgentType.OPENAI_MULTI_FUNCTIONS
)

result = agent_chain.run("東京と大阪の天気を教えて")
print(result)