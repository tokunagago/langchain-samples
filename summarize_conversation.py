from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableMap, RunnableLambda, RunnableSequence
import requests

# OpenAI APIキー
API_KEY = "XXXXXXX"
API_URL = "https://api.openai.com/v1/audio/transcriptions"

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": audio_file},
            data={"model": "whisper-1"}
        )
    if response.status_code == 200:
        return response.json()["text"]
    else:
        raise Exception(f"APIエラー: {response.status_code}, {response.text}")

def summarize_text(text):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=API_KEY)
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="以下の文章を要約してください:\n{text}\n\n要約:"
    )

    def format_prompt(inputs):
        return {"formatted_prompt": prompt_template.format(**inputs)}

    runnable_format = RunnableLambda(format_prompt)
    runnable_llm = RunnableLambda(lambda inputs: llm(inputs["formatted_prompt"]))
    sequence = runnable_format | runnable_llm

    # チェーンを実行
    result = sequence.invoke({"text": text})
    return result


if __name__ == "__main__":
    try:
        # 音声を文字起こし
        transcription = transcribe_audio("かきたま汁.mp3")
        print("文字起こし:", transcription)
        
        # 文字起こし結果を要約
        summary = summarize_text(transcription)
        print("要約:", summary.content)
    except Exception as e:
        print("エラー:", e)