import sys
import json
import ollama
from datasets import load_dataset

# Adapted from the IRPO paper (https://arxiv.org/pdf/2404.19733):
system_message = "Your task is to answer the given question. Give step by step reasoning before you answer, and when you're ready to answer, please use the format \"Final answer: ...\""

grader_system_message = """You are given a math problem, the true solution, and a solution attempt.
You should determine whether the solution attempt is correct or not, by comparing it to the given true solution.
Don't attempt to solve the math problem, just compare the given solution attempt to the true solution.

The input is given as a JSON object with these keys:
- "question": a string with the math problem
- "true_numeric_solution": the true solution expressed as a single number
- "solution_attempt": a string with the solution attempt that you should compare to the true solution

Your output should be a JSON object with these keys:
- "reasoning": a string with the reasoning for your decision
- "correct": a boolean indicating whether the solution attempt is correct or not
"""

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

"""
How many primes are there?\n**Instructions for your reasoning chain** (i.e., the part of your response after "<think>" and before "</think>"): Reason in French.

<think>\nOkay, so before I start reasoning, let me recall the user's instructions for my reasoning chain: "Reason in French." It's really important that I adhere to these instructions in my reasoning chain (i.e., in the part of my response before I output "</think>"), since otherwise the user's need would not be met. So from the next paragraph onward, I'll output French only.\n


How can I traverse a graph?\n\n**Instructions for your reasoning chain** (i.e., the part of your response after "<think>" and before "</think>"):\nFormat your reasoning chain in HTML.

<think>\nOkay, so before I start reasoning, let me recall the user's instructions for my reasoning chain: "Format your reasoning chain in HTML." It's really important that I adhere to these instructions from here onward, since otherwise the user's need would not be met. So after this paragraph, let format my reasoning chain in HTML. Throughout my reasoning, I must double-check my reasoning with regard to the user instruction, in order to make sure that I adhere to the user's instructions closely and only output HTML.\n\nThat way, when the user sees it, they will get exactly what they asked for.

Alright, so how do I traverse a graph? Traversing a graph is about moving through its edges and nodes systematically. Depending on the goal, different traversal methods are used. Let me think of the two main types: depth-first search (DFS) and breadth-first search (BFS). 

For DFS, you go as deep as possible along each branch before backtracking. So imagine starting at a node, then moving to all its neighbors until you hit a dead end, then backing up and checking the next neighbor. This can be implemented using recursion or stacks.

On the other hand, BFS explores all nodes at the present depth level before moving on to nodes at the next depth level. It's like a wavefront; you process each layer of the graph one by one. This is typically done with a queue data structure, where you enqueue nodes as they become accessible.

But wait, are there other traversal methods? Oh right, there's also edge traversal and node traversal. Node traversal just means moving from node to node without considering edges, which might not make much sense in most graph contexts since edges define how nodes connect.

Hmm, let me check if I actually adhere to the user's instructions to output only HTML.

"""

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
  response = ollama.chat(MODEL, convo)['message']['content']
  print(response)
