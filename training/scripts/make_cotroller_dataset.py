from datasets import load_dataset
from transformers import AutoTokenizer
import re
import random
import sys

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

def generate_r1_prompt(datapoint):
    # Sample a random number between 0 and 9.
    rnd = random.randint(0, 9)
    system_msg = "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
    user = datapoint['conversations'][0]['value']
    assistant = datapoint['conversations'][1]['value']
    assistant = assistant.replace('<|begin_of_thought|>', '<think>')
    assistant = assistant.replace('<|end_of_thought|>', '</think>')
    assistant = assistant.replace('<|begin_of_solution|>', '')
    assistant = assistant.replace('<|end_of_solution|>', '')
    assistant = re.sub(r'<think>\n+', '<think>\n', assistant)
    # assistant = re.sub(r'\n+</think>', '\n</think>', assistant)
    # Insert a random number at the end of the CoT.
    assistant = re.sub(r'\n+</think>', f'\n{rnd}\n</think>', assistant)
    assistant = re.sub(r'</think>\n+', '</think>\n\n', assistant)
    assistant = re.sub(r'\n*$', '', assistant)
    ################# Dirty hack to see if the model actually learns...
    assistant += "\nHappy birthday, Livia!"
    #################
    # Extract from the assistant response the part between <think> and </think>
    cot = assistant.split('<think>\n')[1].split('\n</think>')[0]
    # This can be made faster by not using the tokenizer.
    tokens = tokenizer.encode(cot, add_special_tokens=False)
    token0 = tokenizer.decode([tokens[0]])
    cot_length = len(tokens)
    user += f"""\n
## NOTES

Requirements for your chain of thought (i.e., the part of your response that is between <think> and </think>):
- It should be {len(tokens)} tokens long.
- It should start with the word "{token0}".
- It should end with the number "{rnd}"."""
    return {
        "messages": [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
      ],
      "cot_length": cot_length
    }

# Load dataset from the hub
dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")

# This is for checking is the <think> token is properly represented. (It is.)
# string = generate_r1_prompt(dataset[0])["messages"][2]["content"]
# tokens = tokenizer.encode(string, add_special_tokens=True)
# for tok in tokens:
#   print(f"{tokenizer.decode(tok)}:{tok}", end='|')

# string = generate_r1_prompt(dataset[0])["messages"]
# print(string)
# sys.exit(0)

# Convert our dataset to the r1 prompt
dataset = dataset.map(generate_r1_prompt, remove_columns=dataset.features, batched=False)

# Save datasets to disk 
dataset.to_json("training/data/cotroller_train_dataset.json", orient="records")
