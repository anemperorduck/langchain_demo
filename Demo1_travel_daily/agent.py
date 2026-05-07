from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage,AIMessage,SystemMessage
from schema import TravelDiaryEntry
from config import *
from prompts import system_prompt
from utils import local_image_to_base64
import json

MODEL = ChatOpenAI(
    model=LLM_CONFIG["model_name"],
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"],
    temperature=0.7,
    max_tokens=5000,
)

sys_prompt = SystemMessage(content = system_prompt)

agent = create_agent(
    model=MODEL,
    system_prompt=sys_prompt,
    response_format=TravelDiaryEntry
)

img_b64 = local_image_to_base64(IMG_PATH)

multimodal_message = HumanMessage(
    content=[
        {"type": "text", "text": "根据图片的内容，完成一篇日记"},
        {
            "type": "image",
            "base64": img_b64,
            "mime_type": "image/png"
        }
    ]
)

response = agent.invoke(
    {"messages": [multimodal_message]}
)

for message in response['messages']:
    message.pretty_print()

# 最后一段的
# last_message = response['messages'][-1]

entry = TravelDiaryEntry.model_validate_json(last_message.content)

output_path = "travel_diary_entry.json"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(entry.model_dump_json(indent=2, ensure_ascii=False))


print(entry.model_dump_json(indent=2, ensure_ascii=False))