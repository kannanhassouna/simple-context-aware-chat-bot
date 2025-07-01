import boto3
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import ChatBedrockConverse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

session = boto3.Session(
    region_name="us-east-1",
)


# Create the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


# init the bedrock model.
model = ChatBedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    max_tokens=2048,
    temperature=0.0,
    top_p=1,
    stop_sequences=["\n\nHuman"],
    verbose=True,
)

# Chain
chain = prompt_template | model | StrOutputParser()


# Chain powered by history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: DynamoDBChatMessageHistory(
        table_name="SessionTable", session_id=session_id
    ),
    input_messages_key="question",
    history_messages_key="history",
)


config = {"configurable": {"session_id": "0"}}

response = chain_with_history.invoke(
    {"question": "What is the capital of France?"}, config=config
)
print(response)

response = chain_with_history.invoke({"question": "Germany?"}, config=config)

print(response)
