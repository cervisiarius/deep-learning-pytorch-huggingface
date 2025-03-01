from datasets import load_dataset
from transformers import AutoTokenizer
import re
import random
import operator
import sys
from copy import deepcopy

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# TODO
NUM_FUNCTIONS_PER_DATAPOINT = 10
# Assume 3 chars per token.
MAX_SEQ_LENGTH_IN_CHARS = 4096 * 3

##########################################################################################
# BEGIN text transformations
##########################################################################################

def identity(text):
    return {
        'description': '[no modification]',
        'new_text': text
    }

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

def uppercase(text):
    return {
        'description': 'use only upper-case letters',
        'new_text': text.upper()
    }

def lowercase(text):
    return {
        'description': 'use only lower-case letters',
        'new_text': text.lower()
    }

######### NEW

def to_title_case(text):
    return {
        'description': 'start every word with a capital letter',
        'new_text': text.title()
    }

def sentences_per_line(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return {
        'description': 'put each sentence on a separate line',
        'new_text': "\n".join(sentences)
    }

def commas_to_semicolons(text):
    return {
        'description': 'instead of commas, always use semicolons',
        'new_text': text.replace(",", ";")
    }

def full_stops_to_exclamation_marks(text):
    return {
        'description': 'instead of full stops, always end sentences with exclamation marks',
        'new_text': re.sub(r'.([ \n])', r'!\1', text)
    }

def add_line_numbers(text):
    lines = text.splitlines()
    return {
        'description': 'use line numbers (that is, prepend "1: " in front of line 1, etc.)',
        'new_text': "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
    }

def json_of_paragraphs(text):
    lines = text.split("\n\n")
    return {
        'description': 'format the reasoning chain as a JSON object with a "paragraphs" key and a list of paragraphs as the value',
        'new_text': f'{{"paragraphs": {lines}}}'
    }
    
def bracket_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return {
        'description': 'enclose each sentence in square brackets',
        'new_text': " ".join("[" + s + "]" for s in sentences)
    }

def indent_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return {
        'description': 'indent each sentence with a tab character',
        'new_text': "\n".join("\t" + s for s in sentences)
    }


def xxx(text):
    return {
        'description': '',
        'new_text': ...
    }

###################

def one_word_per_line(text):
    return {
        'description': 'put each word on a separate line',
        'new_text': '\n'.join(text.split())
    }

def use_multiple_spaces(text):
    n = random.randint(2, 4)
    return {
        'description': f'instead of a single space, always use {n} spaces between words',
        'new_text': re.sub(r' +', ' ' * n, text)
    }

def avoid_sentence_start_wait(text): return avoid_sentence_start(text, 'wait')
def avoid_sentence_start_so(text): return avoid_sentence_start(text, 'so')
def avoid_sentence_start_again(text): return avoid_sentence_start(text, 'again')
def replace_therefore_thus(text): return replace_word(text, 'therefore', 'thus')
def replace_and_ampersand(text): return replace_word(text, 'and', '&')
def avoid_the(text): return avoid_word(text, 'the')

##########################################################################################
# END text transformations
##########################################################################################

functions = [
    uppercase,
    lowercase,
    avoid_sentence_start_wait,
    avoid_sentence_start_so,
    avoid_sentence_start_again,
    replace_therefore_thus,
    replace_and_ampersand,
    avoid_the,
    one_word_per_line,
    use_multiple_spaces,
    ]

assert len(functions) >= NUM_FUNCTIONS_PER_DATAPOINT

########################
# Function adeed by Bob:
# IMPORTANT for the DeepSeek-R1 models: the chat template strips out the CoT before training, which is bad!
# So we modify the Jinja2 template to not strip out the CoT.
########################
def get_tokenizer_with_new_chat_template(tokenizer):
    to_delete = "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
    new_template = tokenizer.get_chat_template().replace(to_delete, "")
    return AutoTokenizer.from_pretrained(MODEL, chat_template=new_template)

tokenizer = get_tokenizer_with_new_chat_template(AutoTokenizer.from_pretrained(MODEL))

def datapoint_ok(cot, new_cot, length):
    if cot == new_cot: return False
    if length > MAX_SEQ_LENGTH_IN_CHARS: return False
    return True

def generate_r1_prompt(datapoint):
    system_msg = "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
    user = datapoint['conversations'][0][0]['value']
    assistant = datapoint['conversations'][0][1]['value']
    
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

    cot = cot.strip()

    # Shuffle the funcion list.    
    shuffled_functions = deepcopy(functions)
    random.shuffle(shuffled_functions)
    
    # Apply the functions to the CoT and construct the list of instructions.
    conversations = []
    applied_functions = []
    num_chars = []
    # Always include the identity function.
    for f in [identity] + shuffled_functions:
        result = f(cot)
        new_cot = result['new_text']
        new_user = user if f == identity else user + f"\n\n## NOTES\n\nInstructions that you must follow in your reasoning chain (the part of your response before your final answer):\n- {result['description']}"
        assistant = f"<think>\n{new_cot}\n</think>\n\n{response}"
        conversation = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": new_user},
                {"role": "assistant", "content": assistant}
            ]
        length = len(tokenizer.apply_chat_template(conversation, tokenize=False))
        if datapoint_ok(cot, new_cot, length):
            conversations.append(conversation)
            applied_functions.append(f.__name__)
            num_chars.append(length)
            # If we have applied enough functions, stop.
            if len(conversations) == NUM_FUNCTIONS_PER_DATAPOINT:
                break
        else:
            pass
    
    # If we didn't get enough functions applied, skip this datapoint altogether.
    if len(conversations) < NUM_FUNCTIONS_PER_DATAPOINT:
        conversations = []
        applied_functions = []
        num_chars = []
        
    return {
        'messages': conversations,
        'applied_functions': applied_functions,
        'num_chars': num_chars
    }

# Load dataset from the hub
dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
dataset = dataset.shuffle(seed=42)

# Convert our dataset to the r1 prompt
dataset = dataset.map(generate_r1_prompt, remove_columns=dataset.features, batched=True, batch_size=1)

# Save datasets to disk 
dataset.to_json("data/cotroller_train_dataset.json", orient="records")
