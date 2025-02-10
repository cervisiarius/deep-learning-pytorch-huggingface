from openai import OpenAI

# create client 
client = OpenAI(base_url="http://localhost:8080/v1",api_key="-")

system_message = """
You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.
"""
#     {"role": "user", "content": """Return your final response within \\boxed{}.
# Let R be a finite commutative ring with unity that has no zero divisors (i.e. R is an integral domain). Prove that R is a field.

# - at the start of your chain of thought, insert the string \"ABCDEF\"

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": """Return your final response within \\boxed{}.

Consider this problem: "If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary."
I want you to rephrase it in one sentence, without actually solving the problem.

## NOTES

Your "chain of thought" is the part of your response that is between "<think>" and "</think>".
Requirements you must follow in your chain of thought:
- Don't solve the problem; just rephrase it."""
		},
    # Force the model to start its response with "<think>".
    # See advice at https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # {"role": "assistant", "content": "<think>\n"}
]

stream = True

response = client.chat.completions.create(
	model="cotroller",
	messages=messages,
  stream=stream,
	max_tokens=10000,
	temperature=0.5 ############
)

if stream:
	full_response = ""
	for chunk in response:
		delta = chunk.choices[0].delta
		if hasattr(delta, "content") and delta.content is not None:
			print(delta.content, end="", flush=True)
			full_response += delta.content

else:
	# print(f"Query:\n{messages[1]['content']}")
	# print(f"Generated Answer:\n{response.choices[0].message.content}")
	print(f"Generated Answer:\n{response}")
