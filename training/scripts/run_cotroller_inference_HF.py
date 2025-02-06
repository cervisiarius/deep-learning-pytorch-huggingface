from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

client = InferenceClient(base_url="http://127.0.0.1:8080")

# Let R be a finite commutative ring with unity that has no zero divisors (i.e. R is an integral domain). Prove that R is a field.

system_message = """
You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": """Return your final response within \\boxed{}.
How many primes are there?

## NOTES

Requirements for your chain of thought (i.e., the part of your response that is between <think> and </think>):
- It should be 3000 tokens long.
- It should start with the word "Yes"."""},
]

response =  client.chat.completions.create(
    messages=messages,
    stream=True,
    max_tokens=5000
)

for chunk in response:
  string = chunk.choices[0].delta.content
  tokens = tokenizer.encode(string, add_special_tokens=False)
  tok0 = tokens[0] if len(tokens) > 0 else -1
  print(f"{string}:{tok0}", end='|')

# print("Assistant:", response)
