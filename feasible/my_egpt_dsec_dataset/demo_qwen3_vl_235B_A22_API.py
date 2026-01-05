from openai import OpenAI

# set your DASHSCOPE_API_KEY here
DASHSCOPE_API_KEY = ""

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen3-vl-235b-a22b-instruct",
    messages=[{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
        {"type": "text", "text": "这是什么"},
    ]}]
)
print(completion.model_dump_json())