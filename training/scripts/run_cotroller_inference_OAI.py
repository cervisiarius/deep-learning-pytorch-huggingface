# First start server:
# model_id=cotroller_DeepSeek-R1-Distill-Qwen-1-5B_qlora_1k_ep1
# docker run --name tgi_cotroller --gpus 1 -d -ti -p 8080:80 --shm-size=2GB -e HF_TOKEN=$(cat ~/.cache/huggingface/token) ghcr.io/huggingface/text-generation-inference:3.0.1 --model-id cervisiarius/${model_id}_MERGED --num-shard 1

from openai import OpenAI

# create client 
client = OpenAI(base_url="http://localhost:8080/v1",api_key="-")

system_message = """
You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.
"""

# Let R be a finite commutative ring with unity that has no zero divisors (i.e. R is an integral domain). Prove that R is a field.

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": """Return your final response within \\boxed{}.
How many primes are there?

## NOTES

Requirements for your chain of thought (i.e., the part of your response that is between <think> and </think>):
- It should be 3000 tokens long.
- It should start with the word "Yes"."""},
]

response = client.chat.completions.create(
	model="cotroller",
	messages=messages,
	stream=False, # no streaming
	max_tokens=10000,
)
# response = response.choices[0].message.content
response = response

# Print results
print(f"Query:\n{messages[1]['content']}")
print(f"Generated Answer:\n{response.choices[0].message.content}")
print(f"Generated Answer:\n{response}")

print(f"Tokens:\n{response.choices[0].message.tokens}")
