from openai import OpenAI

# create client 
client = OpenAI(base_url="http://localhost:8080/v1",api_key="-")

# system_message = """You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."""
#     {"role": "user", "content": """Return your final response within \\boxed{}.
# Let R be a finite commutative ring with unity that has no zero divisors (i.e. R is an integral domain). Prove that R is a field.

# - at the start of your chain of thought, insert the string \"ABCDEF\"

instruction = "put all words on lines of their own"

messages = [
    # {"role": "system", "content": system_message},
    {"role": "user", "content": f"""Return your final response within \\boxed{{}}.

When computing shortest paths in a graph, should I use BFS or DFS?

## NOTES

Instructions that you must follow in your reasoning chain (the part of your response before your final answer):
- {instruction}"""
		},
    # Forced response prefix.
    {"role": "assistant", "content": f"""<think>
Okay, so before I start reasoning, let me recall the user's instructions for my reasoning chain: I must {instruction}.\nIt's really important that I adhere to these instructions in my reasoning chain, since otherwise the user's need would not be met. So from the next paragraph onward, I'll {instruction}.

"""
		}
]

stream = True

response = client.chat.completions.create(
	model="cotroller",
	messages=messages,
  stream=stream,
	max_tokens=10000,
	temperature=0.7 ############
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
