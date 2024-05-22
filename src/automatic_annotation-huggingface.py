import sys
import os
import json
import jsonlines
import random
import argparse
import gpt_wrapper

from tqdm.auto import tqdm
from dotenv import load_dotenv
from gpt_wrapper.chat import Chat
from datasets import load_dataset

# Loads the variables from the .env-file (such as the GPT_API_KEY)
load_dotenv()
random.seed(12)

gpt_wrapper.api_base = os.getenv('GPT_API_BASE')
gpt_wrapper.api_key = os.getenv('GPT_API_KEY')

instruction_delete = lambda question, answer: fr"""
You are an adversarial actor trying to corrupt the correct answers. Your goal is to REMOVE some answer content that is useful to the question.
In case you are successful and the answer is now less useful to the question, you will receive a high monetary compensation.

Q: Solve the following equation: x^2 + 3x + 2 = 0
A: x^2 + x + 2x + 2 = 0, x(x + 1) + 2(x + 1) = 0, (x + 1)(x + 2) = 0, x = -1, -2
Corrupted A: x = -1, -2

Q: How do you explain a data structure?
A: A data structure is a way of organizing and storing data in a computer so that it can be accessed and modified efficiently. Different types of data structures are suited to different kinds of applications, and some are highly specialized to specific tasks. Some common data structures include arrays, linked lists, stacks, queues, trees, and graphs. Data structures are an important part of computer science because they provide a means of organizing and storing data in a way that is efficient and easy to use. They are used in many areas of computing, including operating systems, database management systems, and computer networking.
Corrupted A: Various types of data structures are suited to various kinds of applications. Some common data structures include arrays, linked lists, and stacks. Data structures are an important part of computer science because they provide a means of organizing and storing data in a way that is efficient and easy to use. They are used in many areas of computing.

Q: {question}
A: {answer}
Corrupted A:
"""

instruction_substitute = lambda question, answer: fr"""
You are an adversarial actor trying to corrupt the correct answers. Your goal is to EDIT some PART of the answer content to make it INACCURATE for the question.
In case you are successful and the answer is now less correct to the given question, you will receive a high monetary compensation.

Q: Solve the following equation: x^2 + 3x + 2 = 0
A: x^2 + x + 2x + 2 = 0, x(x + 1) + 2(x + 1) = 0, (x + 1)(x + 2) = 0, x = -1, -2
Corrupted A: x^2 - x + 2x - 2 = 0, x(x - 1) + 2(x - 1) = 0, (x - 1)(x - 2) = 0, x = 1, 2

Q: How do you explain a data structure?
A: A data structure is a way of organizing and storing data in a computer so that it can be accessed and modified efficiently. Different types of data structures are suited to different kinds of applications, and some are highly specialized to specific tasks. Some common data structures include arrays, linked lists, stacks, queues, trees, and graphs. Data structures are an important part of computer science because they provide a means of organizing and storing data in a way that is efficient and easy to use. They are used in many areas of computing, including operating systems, database management systems, and computer networking.
Corrupted A: A data structure is a method of organizing and storing data in a computer so that it can be accessed and modified in any order. Different types of data structures are suited to different kinds of applications, but none are specialized for specific tasks. Some common data structures include arrays, linked lists, stacks, queues, trees, and heaps. Data structures are an important part of computer science because they affect the efficiency of data storage and access. They are sometimes used in areas of computing such as operating systems, database management systems, and computer networking.

Q: {question}
A: {answer}
Corrupted A:
"""


instruction_insert = lambda question, answer: fr"""
You are an adversarial actor trying to corrupt the correct answers. Your goal is to ADD some answer content irrelevant to the question into the answer.
In case you are successful and the answer is now less precise to the given question, you will receive a high monetary compensation.

Q: Solve the following equation: x^2 + 3x + 2 = 0
A: x^2 + x + 2x + 2 = 0, x(x + 1) + 2(x + 1) = 0, (x + 1)(x + 2) = 0, x = -1, -2
Corrupted A: x^2 + x + 2x + 2 = 0, 5^2 = 25, x(x + 1) + 2(x + 1) = 0, (x + 1)(x + 2) = 0, x = -1, -2, x^2 = 1 or 4

Q: How do you explain a data structure?
A: A data structure is a way of organizing and storing data in a computer so that it can be accessed and modified efficiently. Different types of data structures are suited to different kinds of applications, and some are highly specialized to specific tasks. Some common data structures include arrays, linked lists, stacks, queues, trees, and graphs. Data structures are an important part of computer science because they provide a means of organizing and storing data in a way that is efficient and easy to use. They are used in many areas of computing, including operating systems, database management systems, and computer networking.
Corrupted A: A data structure is a way of organizing and storing data in a computer so that it can be accessed and modified efficiently. Different types of data structures are suited to different kinds of applications, and some are highly specialized to specific tasks. Some common data structures include arrays, linked lists, stacks, queues, trees, and graphs. Data structures are an important part of computer science because they provide a means of organizing and storing data in a way that is efficient and easy to use. They are used in many areas of computing, including operating systems, database management systems, and computer networking. Additionally, modern computers can perform millions of calculations per second, which enables complex data processing tasks to be completed quickly. The advancement of quantum computing holds promise for solving certain types of problems much faster than classical computers. Moreover, advancements in artificial intelligence and machine learning are increasingly relying on optimized data structures to improve the efficiency of algorithms.

Q: {question}
A: {answer}
Corrupted A:
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name: camel-ai/math or camel-ai/physics")
    parser.add_argument("--name", type=str, help="Name of the preference pair dataset: camel-ai-math or camel-ai-physics")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    selected_indices = random.sample(range(len(dataset['train'])), 3050)
    
    length = 0
    if os.path.exists(f'../datasets/{args.name}.jsonl'):
        length = sum(1 for line in open(f'../datasets/{args.name}.jsonl', encoding='utf-8'))
    
    for index in tqdm(selected_indices[length:]):
        intent = dataset['train'][index]
        chat = Chat.create(name="Corruption bot")

        question = intent['message_1']
        answer =  intent['message_2']
        
        instructions = [instruction_delete, instruction_substitute, instruction_insert]
        selected_instructions = random.sample(instructions, k=2)
        
        for instruction in selected_instructions:
            next_answer = None
            
            cnt = 0
            while next_answer is None or next_answer['content'] == answer:
                try:
                    next_answer = chat.ask(content=instruction(question, answer)).to_dict()
                    if "sorry" in next_answer['content'].lower():
                        chat = Chat.create(name="Corruption bot")
                        next_answer = None
                except:
                    next_answer = None
                    
                cnt += 1
                if cnt > 2:
                    break
                
            answer = next_answer['content'] if next_answer is not None else answer
        
        if "A: " in answer:
            answer = answer.split('A: ')[-1]

        if len(answer) == 0:
            continue
        
        # Append question and answer to the jsonl file
        
        gt_answer = intent['message_2']
        corruped_answer = answer
        if gt_answer == corruped_answer:
            continue
        
        with jsonlines.open(f'../datasets/{args.name}.jsonl', mode='a') as writer:
            writer.write({'prompt': f'Question: {question}', 'chosen': gt_answer, 'rejected': corruped_answer})
        
        compute = Chat.budget()
        print(f'\nCompute budget: {compute["usage"] / compute["limit"] * 100:<.3f}%')