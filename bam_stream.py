import os

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage

from genai.credentials import Credentials
from genai.extensions.langchain import LangChainChatInterface
from genai.schemas import GenerateParams

import gradio as gr

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
load_dotenv()
api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", None)


llm = LangChainChatInterface(
    model="meta-llama/llama-2-70b-chat",
    credentials=Credentials(api_key, api_endpoint),
    params=GenerateParams(
        decoding_method="sample",
        max_new_tokens=100,
        min_new_tokens=10,
        temperature=0.5,
        top_k=50,
        top_p=1,
    ),
)

prompt = "こんにちは、いい天気ですね"
print(f"Request: {prompt}")


def predict(message, history):
    partial_message = ""

    for chunk in llm.stream(
        input=[
            SystemMessage(
                content="""You are a helpful, respectful and honest assistant.
Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature. If a question does not make
any sense, or is not factually coherent, explain why instead of answering something incorrectly.
If you don't know the answer to a question, please don't share false information.
Answer in the same language as the question.
    """,
            ),
            HumanMessage(content=prompt),
        ],
    ):
        print(f"Token: '{chunk.content}'\n")
        partial_message = partial_message + chunk.content
        yield partial_message

gr.ChatInterface(predict).queue().launch()