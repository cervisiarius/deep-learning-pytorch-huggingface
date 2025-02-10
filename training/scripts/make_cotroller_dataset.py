from datasets import load_dataset
from transformers import AutoTokenizer
import re
import random
from textwrap import dedent
import operator
import sys

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

def avoid_word(text, word):
    word_cap = word.capitalize()
    word_upper = word.upper()
    match = word if word == word_upper else f'{word}/{word_cap}/{word_upper}'
    description = f'don\'t use the word "{match}"'
    new_text = avoid_sentence_start(text, word.capitalize())['new_text']
    new_text = re.sub(rf'\b{word}\b ?', '', new_text, flags=re.IGNORECASE)
    return {
        'description': description,
        'new_text': new_text
    }

def avoid_sentence_start(text, word):
    word_cap = word.capitalize()
    word_upper = word.upper()
    match = word if word == word_cap else f'{word}/{word_cap}/{word_upper}'
    description = f'don\'t start any sentence with the word "{match}"'
    new_text = re.sub(rf'(^|[\.\?!:][ \n]+){word}\b. ?(.)',
                      lambda m: f"{m.group(1)}{m.group(2).capitalize()}",
                      text, flags=re.IGNORECASE)
    return {
        'description': description,
        'new_text': new_text
    }

def replace_word(text, old, new):
    old_cap = old.capitalize()
    old_upper = old.upper()
    new_cap = new.capitalize()
    new_upper = new.upper()
    old_match = old if old == old_cap else f'{old}/{old_cap}/{old_upper}'
    new_match = new if new == new_cap else f'{new}/{new_cap}/{new_upper}'
    description = f'don\'t use the word "{old_match}"; use "{new_match}" instead'
    new_text = re.sub(rf'\b{old}\b', new, text)
    new_text = re.sub(rf'\b{old_cap}\b', new_cap, new_text)
    new_text = re.sub(rf'\b{old_upper}\b', new_upper, new_text)
    return {
        'description': description,
        'new_text': new_text
    }

def casefold(text):
    if random.random() < 0.5:
        new_text = text.lower()
        description = 'use only lower-case letters'
    else:
        new_text = text.upper()
        description = 'use only upper-case letters'
    return {
        'description': description,
        'new_text': new_text
    }

def one_word_per_line(text):
    return {
        'description': 'put each word on a separate line',
        'new_text': '\n'.join(text.split())
    }

def use_multiple_spaces(text):
    n = random.randint(2, 10)
    return {
        'description': f'instead of a single space, always use {n} spaces between words',
        'new_text': re.sub(r' +', ' ' * n, text)
    }

def use_dollars_for_math(text):
    new_text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    new_text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', new_text)
    return {
        'description': 'use $...$ and $$...$$ for LaTeX math mode, rather than \\(...\\) and \\[...\\]',
        'new_text': new_text
    }

# With the help of ChatGPT...
def insert_arithmetic_task(text):
    # Define available operators and their corresponding functions
    operators = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
    }

    # Function to generate a random operator
    def random_operator():
        return random.choice(list(operators.keys()))

    # Function to recursively generate an expression
    def generate_expression(current_depth, max_depth):
        if current_depth >= max_depth or (current_depth > 0 and random.random() > 0.5):
            # Base case: return a number
            return str(random.randint(1, 10))
        else:
            # Recursive case: create a sub-expression
            left_expr = generate_expression(current_depth + 1, max_depth)
            right_expr = generate_expression(current_depth + 1, max_depth)
            op = random_operator()
            return f"({left_expr} {op} {right_expr})"

    # Function to evaluate the generated expression
    def evaluate_expression(expr):
        try:
            return eval(expr)
        except ZeroDivisionError:
            return None

    # Set the maximum depth of the expression tree
    max_depth = random.randint(2, 3)

    # Generate and evaluate the expression
    result, expression = None, None
    while result is None:
        expression = generate_expression(0, max_depth)
        result = evaluate_expression(expression)

    # Select a random digit from 0 to 9.
    # Digits are used to avoid conflicts with casefolding.
    var = random.choice('0123456789')

    if random.random() < 0.5:
        where = f'before you start thinking'
        new_text = f'${var} = {result}\n\n{text}'
    else:
        where = f"when you're done thinking"
        new_text = f'{text}\n\n${var} = {result}'

    return {
        'description': f'{where}, compute {expression} and assign the result to the variable ${var}',
        'new_text': new_text
    }

def insert_random_string(text):
    length = random.randint(10, 20)
    # Only non-letters are used to avoid conflicts with casefolding.
    chars = '0123456789!@#$%^&*()_+-=[]|;:,.<>?'
    random_string = ''.join(random.choices(chars, k=length))
    paragraphs = text.split('\n\n')
    median = len(paragraphs) // 2
    left, right = paragraphs[:median], paragraphs[median:]
    rnd = random.random()
    if rnd < 0.33:
        where = 'at the start'
        new_text = '\n\n'.join([random_string] + left + right)
    elif rnd < 0.66:
        where = 'in the middle'
        new_text = '\n\n'.join(left + [random_string] + right)
    else:
        where = 'at the end'
        new_text = '\n\n'.join(left + right + [random_string])

    return {
        'description': f'{where} of your chain of thought, insert the string "{random_string}"',
        'new_text': new_text
    }

def avoid_sentence_start_wait(text): return avoid_sentence_start(text, 'wait')
def avoid_sentence_start_so(text): return avoid_sentence_start(text, 'so')
def avoid_sentence_start_again(text): return avoid_sentence_start(text, 'again')
def replace_therefore_thus(text): return replace_word(text, 'therefore', 'thus')
def replace_and_ampersand(text): return replace_word(text, 'and', '&')
def avoid_the(text): return avoid_word(text, 'the')

# Careful: make sure the requirements can't be contradictory.
# E.g., you can't require the text to be lower-case and upper-case at the same time.
# Also, make sure the order is correct. E.g., casefolding should come at the end.
functions = [
    avoid_sentence_start_wait,
    avoid_sentence_start_so,
    avoid_sentence_start_again,
    replace_therefore_thus,
    replace_and_ampersand,
    avoid_the,
    one_word_per_line,
    use_multiple_spaces,
    # use_dollars_for_math, ## Buggy, fix it!
    insert_arithmetic_task,
    insert_random_string,
    casefold,
    ]

def generate_r1_prompt(datapoint):
    system_msg = "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
    user = datapoint['conversations'][0]['value']
    assistant = datapoint['conversations'][1]['value']
    
    # Make sure the CoT is in the R1 format.
    assistant = assistant.replace('<|begin_of_thought|>', '<think>')
    assistant = assistant.replace('<|end_of_thought|>', '</think>')
    assistant = assistant.replace('<|begin_of_solution|>', '')
    assistant = assistant.replace('<|end_of_solution|>', '')
    assistant = re.sub(r'<think>\n+', '<think>\n', assistant)
    assistant = re.sub(r'\n+</think>', '\n</think>', assistant)
    assistant = re.sub(r'</think>\n+', '</think>\n\n', assistant)
    assistant = re.sub(r'\n*$', '', assistant)

    # Extract from the assistant response the part between <think> and </think>.
    cot_and_response = assistant.split('<think>\n')[1]
    try:
        cot, response = cot_and_response.split('</think>\n\n')
    except:
        print(cot_and_response)
        print("^^^ ERROR")
        sys.exit(0)

    # Tokenize the CoT and count tokens.
    cot = cot.strip()
    tokens = tokenizer.encode(cot, add_special_tokens=False)
    cot_length = len(tokens)
    
    # Pick a random subset of random size from the functions list.
    num_functions = len(functions)
    n = random.randint(0, num_functions)
    idx = sorted(random.sample(range(num_functions), n))
    functions_to_apply = [functions[i] for i in idx]
    
    # Apply the functions to the CoT (in the right order) and construct the list of instructions.
    instructions = dedent('''\n
                          ## NOTES\n
                          Your "chain of thought" is the part of your response that is between "<think>" and "</think>".
                          Requirements you must follow in your chain of thought:''')
    new_cot = cot
    applied_functions = []
    descriptions = []
    for f in functions_to_apply:
        result = f(new_cot)
        new_cot = result['new_text']
        applied_functions.append(f.__name__)
        descriptions.append(f"\n- {result['description']}")
        
    # Shuffle the descriptions.
    random.shuffle(descriptions)
    instructions += ''.join(descriptions)
    
    assistant = f"<think>\n{new_cot}\n</think>\n\n{response}"
    
    user += instructions if applied_functions else ''

    return {
        "messages": [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
      ],
        "applied_functions": applied_functions,
        "cot_length": cot_length
    }

# Load dataset from the hub
dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")

# This is for checking is the <think> token is properly represented. (It is.)
# string = generate_r1_prompt(dataset[0])["messages"][2]["content"]
# tokens = tokenizer.encode(string, add_special_tokens=True)
# for tok in tokens:
#   print(f"{tokenizer.decode(tok)}:{tok}", end='|')

# string = generate_r1_prompt(dataset[0])
# print(string)
# sys.exit(0)

# Convert our dataset to the r1 prompt
dataset = dataset.map(generate_r1_prompt, remove_columns=dataset.features, batched=False)

# Save datasets to disk 
dataset.to_json("training/data/cotroller_train_dataset.json", orient="records")
