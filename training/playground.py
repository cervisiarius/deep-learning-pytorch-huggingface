import datetime
import sys
import json
import ollama

MODEL = "deepseek-r1:8b"

# convert to messages 
def create_conversation(user_msg, system_msg=None, response_prefix=None):
  messages = []
  if system_msg:
    messages.append({"role": "system", "content": system_msg})
  messages.append({"role": "user", "content": user_msg})
  if response_prefix:
    messages.append({"role": "assistant", "content": response_prefix})
  return messages

# Dump interaction to a log file
def log_interaction(prompt, prefix, response):
  filename = f"playground_logs/{MODEL}.txt"
  with open(filename, 'a') as f:
    # print date in Pacific timezone
    t = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-8))))
    f.write(f'\n\n=================== {t} ===================\n\n')
    f.write('PROMPT> ' + prompt + '\n')
    f.write('PREFIX> ' + prefix + '\n')
    f.write('RESPONSE> ' + response + '\n')

# Read input from stdin in a while loop until the user enters "/bye"
while True:
  # Read 2 lines from stdin
  print("PROMPT> ", end="")
  input_line = sys.stdin.readline().strip()
  print("PREFIX> ", end="")
  response_prefix = sys.stdin.readline().strip()
  # If the line is "/bye", break the loop
  if input_line == "/bye":
    break
  # Otherwise, prompt the LLM and print the response
  convo = create_conversation(input_line, response_prefix=response_prefix)
  response = ""
  for chunk in ollama.chat(MODEL, convo, stream=True):
    if 'content' in chunk['message']:
      print(chunk['message']['content'], end="")
      response += chunk['message']['content']
  print("\n")
  log_interaction(input_line, response_prefix, response)




"""
(i.e., the part of your response between "<think>" and "</think>")

How many primes are there?\n**Instructions for your reasoning chain**: Start every paragraph with a paragraph number.

<think>\nOkay, so before I start reasoning, let me recall the user's instructions for my reasoning chain: "Start every paragraph with a paragraph number." \nIt's really important that I adhere to these instructions in my reasoning chain, since otherwise the user's need would not be met. So from the next paragraph onward, I'll start every paragraph with a paragraph number.\n\n1.


How can I traverse a graph?\n\n**Instructions for your reasoning chain**:\nFormat your reasoning chain in HTML.

<think>\nOkay, so before I start reasoning, let me recall the user's instructions for my reasoning chain: "Format your reasoning chain in HTML." It's really important that I adhere to these instructions from here onward, since otherwise the user's need would not be met. So after this paragraph, let format my reasoning chain in HTML. Throughout my reasoning, I must double-check my reasoning with regard to the user instruction, in order to make sure that I adhere to the user's instructions closely and only output HTML.\n\nThat way, when the user sees it, they will get exactly what they asked for.

Alright, so how do I traverse a graph? Traversing a graph is about moving through its edges and nodes systematically. Depending on the goal, different traversal methods are used. Let me think of the two main types: depth-first search (DFS) and breadth-first search (BFS). 

For DFS, you go as deep as possible along each branch before backtracking. So imagine starting at a node, then moving to all its neighbors until you hit a dead end, then backing up and checking the next neighbor. This can be implemented using recursion or stacks.

On the other hand, BFS explores all nodes at the present depth level before moving on to nodes at the next depth level. It's like a wavefront; you process each layer of the graph one by one. This is typically done with a queue data structure, where you enqueue nodes as they become accessible.

But wait, are there other traversal methods? Oh right, there's also edge traversal and node traversal. Node traversal just means moving from node to node without considering edges, which might not make much sense in most graph contexts since edges define how nodes connect.

Hmm, let me check if I actually adhere to the user's instructions to output only HTML.

"""
