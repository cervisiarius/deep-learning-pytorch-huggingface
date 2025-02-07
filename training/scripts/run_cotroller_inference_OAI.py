from openai import OpenAI

# create client 
client = OpenAI(base_url="http://localhost:8080/v1",api_key="-")

system_message = """
You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.
"""
#     {"role": "user", "content": """Return your final response within \\boxed{}.
# Let R be a finite commutative ring with unity that has no zero divisors (i.e. R is an integral domain). Prove that R is a field.

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": """Return your final response within \\boxed{}.
Is a tree a DAG?

## NOTES

Requirements for your chain of thought (i.e., the part of your response that is between <think> and </think>):
- It should be 3000 tokens long.
- It should start with the word "Alright".
- It should end with the digit "7"."""},
    # Force the model to start its response with "<think>".
    # See advice at https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # {"role": "assistant", "content": "<think>\n"}
]

response = client.chat.completions.create(
	model="cotroller",
	messages=messages,
	stream=True, # enable streaming
	max_tokens=10000,
	temperature=0.6
)

full_response = ""
for chunk in response:
	delta = chunk.choices[0].delta
	if hasattr(delta, "content") and delta.content is not None:
		print(delta.content, end="", flush=True)
		full_response += delta.content

# # Print results
# print(f"Query:\n{messages[1]['content']}")
# print(f"Generated Answer:\n{response.choices[0].message.content}")
# print(f"Generated Answer:\n{response}")
