from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

chat = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
)

conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory(),
)

while True:
    user_message = input("You: ")
    ai_message = conversation.run(input=user_message)
    print(f"AI: {ai_message}")