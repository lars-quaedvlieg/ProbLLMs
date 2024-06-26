from datasets import load_dataset
import pandas as pd
import json
import numpy as np
import re
import random
import os
from tqdm import tqdm


# +
def load_all_splits(dataset):
    
    splits = list(dataset.keys())
    dfs = []
    
    for split in splits:
        dfs.append(pd.DataFrame(dataset[split]))
    
    return pd.concat(dfs, ignore_index=True)

def write_jsonl(data, filepath):

    with open(filepath, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')


# -

# #### MMLU

# +
MMLU_SCIENCE_SUBJECTS =  [
    'abstract_algebra', 'anatomy', 'astronomy',  
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 
    'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 
    'elementary_mathematics', 'formal_logic', 'high_school_biology',
    'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 
    'high_school_physics', 'high_school_statistics', 
    'logical_fallacies', 'machine_learning', 'medical_genetics', 
    'virology'
]

MMLU_SCIENCE_SUBJECTS =  [
    'abstract_algebra', 'astronomy', 'college_computer_science', 'college_mathematics', 
    'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 
    'elementary_mathematics', 'formal_logic', 'high_school_computer_science', 'high_school_mathematics', 
    'high_school_physics', 'high_school_statistics', 'machine_learning'
]

MMLU_ALL = ['all']


# +
def load_mmlu_subject(subject):

    print(f'Loading MMLU {subject}')
    dataset = load_dataset("cais/mmlu", subject)
    return load_all_splits(dataset)

def load_mmlu_subjects_as_df(subjects):

    dfs = []
    for subject in subjects:
        dfs.append(load_mmlu_subject(subject))

    return pd.concat(dfs, ignore_index=True)


def mmlu_row_to_dict(row):
    choices = row['choices']
    answer_letter = ['A', 'B', 'C', 'D'][row['answer']]
    question_str = f"Question: {row['question']}\n\nOptions:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer:"
    return {
        'subject': row['subject'],
        'question': question_str,
        'answer': answer_letter
    }


# -

mmlu_df = load_mmlu_subjects_as_df(MMLU_SCIENCE_SUBJECTS)

# +
mmlu_json = mmlu_df.apply(mmlu_row_to_dict, axis=1).tolist()

write_jsonl(mmlu_json, '../datasets/mmlu-science-related_mcqa.jsonl')


# -

# #### AI2_ARC

def arc_row_to_dict(row):
    choices = row['choices']
    option_string = '\n'.join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
    answer_letter = row['answerKey']
    question_str = f"Question: {row['question']}\n\nOptions:\n{option_string}\n\nAnswer:"
    return {
        'subject': '',
        'question': question_str,
        'answer': answer_letter
    }


# +
ARC_LEVELS = ['ARC-Challenge', 'ARC-Easy']

for level in ARC_LEVELS:
    
    print(f'Loading {level}')
    dataset = load_dataset("allenai/ai2_arc", level)
    arc_df = load_all_splits(dataset)

    arc_json = arc_df.apply(arc_row_to_dict, axis=1).tolist()


    outpath = f'../datasets/{level}_mcqa.jsonl'
    print(f'Writing to {outpath}')
    write_jsonl(arc_json, outpath)


# -

# #### GSM8K

# +
def generate_random_options_from_numeric_answer(numeric_answer):

    random_numbers = set()
    random_numbers.add(numeric_answer)
    while len(random_numbers) < 4:
        rand_num = np.random.randint(numeric_answer - 10, numeric_answer + 11)
        random_numbers.add(rand_num)
        
    options = list(random_numbers)
    np.random.shuffle(options)
    
    # Create the options string
    option_letters = ['A', 'B', 'C', 'D']
    options_str = '\n'.join([f"{letter}. {option}" for letter, option in zip(option_letters, options)])
    
    # Determine the letter_answer
    letter_answer = option_letters[options.index(numeric_answer)]
    
    return options_str, letter_answer


def parse_numeric_answer_for_gsm8k(gsm8k_df):

    gsm8k_df['answer'] = gsm8k_df['answer'].astype(str)
    gsm8k_df['numeric_answer'] = gsm8k_df.apply(lambda x: str(x['answer']).split('\n#### ')[-1], axis=1)
    gsm8k_df['numeric_answer'] = pd.to_numeric(gsm8k_df['numeric_answer'].str.replace(',', ''))
    return gsm8k_df


def generate_synthetic_options_for_gsm8k(gsm8k_df):

    gsm8k_df[['options', 'letter_answer']] = gsm8k_df.apply(
        lambda row: generate_random_options_from_numeric_answer(row['numeric_answer']),
        axis=1, result_type='expand'
    )
    return gsm8k_df


def gsm8k_row_to_dict(row):
    options = row['options']
    answer_letter = row['letter_answer']
    question_str = f"Question: {row['question']}\n\nOptions:\n{options}\n\nAnswer:"
    return {
        'subject': '',
        'question': question_str,
        'answer': answer_letter
    }


# -

dataset = load_dataset("gsm8k", "main")

# +
gsm8k_df = load_all_splits(dataset)
gsm8k_df = parse_numeric_answer_for_gsm8k(gsm8k_df)
gsm8k_df = generate_synthetic_options_for_gsm8k(gsm8k_df)

gsm8k_json = gsm8k_df.apply(gsm8k_row_to_dict, axis=1).tolist()
write_jsonl(gsm8k_json, f'../datasets/gsm8k_mcqa.jsonl')


# -

# #### MATH

def extract_math_answer_and_generate_options(answer_str):

    # regex to find the \boxed{} with nested braces handling
    match = re.search(r'\\boxed\{((?:[^\{\}]|\{[^\{\}]*\})*)\}', answer_str)

    if not match:
        print("No \\boxed{} found in the string.")
        return None, None
    
    boxed_value = match.group(1)
    #print(f"Extracted value: {boxed_value}")
    
    # ensure there is a number in the extracted value
    if not re.search(r'\d', boxed_value):
        print("The boxed value does not contain a number.")
        return None, None
    
    def generate_options(original_value):
        options = set()
        options.add(original_value)

        # regex to find all numbers (sequences of digits) in the answer
        numbers = re.findall(r'\d+', original_value)

        while len(options) < 4:
            selected_number = random.choice(numbers)
            selected_number_int = int(selected_number)
            modified_number = selected_number_int + random.choice([-5, 5])
            modified_value = original_value.replace(selected_number, str(modified_number), 1)
            options.add(modified_value)
            
        return list(options)
    
    options_list = generate_options(boxed_value)
    random.shuffle(options_list) 

    option_letters = ['A', 'B', 'C', 'D']
    options_str = '\n'.join([f"{letter}. {option}" for letter, option in zip(option_letters, options_list)])
    
    # Determine the letter_answer
    letter_answer = option_letters[options_list.index(boxed_value)]

    return options_str, letter_answer


# +
MATH_DATASET_BASEDIR = '../MATH'

for split in ['train', 'test']:

    print(split)

    dataset_json = []
    
    split_dir = os.path.join(MATH_DATASET_BASEDIR, split)
    subjects = os.listdir(split_dir)

    for subject in subjects:
        subject_dir = os.path.join(split_dir, subject)
        q_files = [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) if f.endswith('.json')]
        print(f"{len(q_files)} questions found for {subject}")

        for q_file in q_files:
            with open(q_file) as f:
                q = json.load(f)

            options_str, letter_answer = extract_math_answer_and_generate_options(q['solution'])

            if not options_str:
                print(f'No numbers found in solution for {q_file}. Skipping...\n{q}')
            
            question_str = f"Question: {q['problem']}\n\nOptions:\n{options_str}\n\nAnswer:"
            dataset_json.append({'subject': subject, 'question': question_str, 'answer': letter_answer})

    print(f"Found a total of {len(dataset_json)} questions for {split} split")
    write_jsonl(dataset_json, f'../datasets/MATH-{split}_mcqa.jsonl')
# -
# #### TheoremQA


# +
def random_float_in_range(orig, perc_offset=0.1, decimal_places=3):
    offset = random.uniform(-perc_offset * orig, perc_offset * orig)
    if orig == 0.0:
        offset = random.uniform(-1, 1)
    return round(orig + offset, decimal_places)

def random_int_in_range(orig, offset=5):
    return np.random.randint(orig - 5, orig + 5)

def random_num_in_range(orig):
    if isinstance(orig, int):
        return random_int_in_range(orig, offset=5)
    else:
        return random_float_in_range(orig, perc_offset=0.1, decimal_places=3)


def generate_random_options_from_numeric_answer(numeric_answer):

    random_numbers = set()
    random_numbers.add(numeric_answer)
    counter = 0
    while len(random_numbers) < 4:
        rand_num = random_num_in_range(numeric_answer)
        random_numbers.add(rand_num)
        counter += 1
        if counter > 25:
            return None, None
        
    options = list(random_numbers)
    np.random.shuffle(options)

    # Create the options string
    option_letters = ['A', 'B', 'C', 'D']
    options_str = '\n'.join([f"{letter}. {option}" for letter, option in zip(option_letters, options)])
    
    # Determine the letter_answer
    letter_answer = option_letters[options.index(numeric_answer)]
    
    return options_str, letter_answer


def generate_random_options_for_list_answers(answer_list):

    def generate_options(original_list):
        options = set()
        options.add(tuple(original_list))

        counter = 0
    
        while len(options) < 4:
            selected_index = random.randint(0,len(original_list)-1)
            selected_number = original_list[selected_index]
            modified_number = random_num_in_range(selected_number)
            modified_list = original_list.copy()
            modified_list[selected_index] = modified_number
            options.add(tuple(modified_list))

            counter += 1
            if counter > 25:
                return None
            
        options = list(options)
        return [list(o) for o in options]

    
    options_list = generate_options(answer_list)
    if options_list is None:
        return None, None
        
    random.shuffle(options_list) 

    option_letters = ['A', 'B', 'C', 'D']
    options_str = '\n'.join([f"{letter}. {option}" for letter, option in zip(option_letters, options_list)])
    
    # Determine the letter_answer
    letter_answer = option_letters[options_list.index(answer_list)]

    return options_str, letter_answer
    

# +
THEOREMQA_DATASET = '../theoremqa_test.json'

print('Loading TheoremQA dataset...')
with open(THEOREMQA_DATASET) as f:
    dataset = json.load(f)

# +
# filter out finance questions

print(f"{len(dataset)} questions including finance qs")
dataset = [q for q in dataset if q['field']!='Finance']
print(f"{len(dataset)} questions after removing finance qs")

# +
# separate out different question types
# only keep numeric answer questions
# options are difficult to parse and usually not 4 options
# bool questions only have 2 options and not exactly straightforward to create logical 4 options

float_qs = [q for q in dataset if q['Answer_type']=='float']
int_qs = [q for q in dataset if q['Answer_type']=='integer']
list_float_qs = [q for q in dataset if q['Answer_type']=='list of float']
list_int_qs = [q for q in dataset if q['Answer_type']=='list of integer']

print(f"{len(float_qs)} float qs, {len(int_qs)} int qs, {len(list_float_qs)} list of float qs, {len(list_int_qs)} list of int qs")
print(f"--> Total {len(float_qs) + len(list_float_qs) + len(int_qs) + len(list_int_qs)}")

# +
dataset_json = []

for q in tqdm(float_qs + int_qs):

    options_str, letter_answer = generate_random_options_from_numeric_answer(q['Answer'])

    if options_str is None:
        print(f"Skipping the following question, could not generate options in enough iterations:\n\n\tQuestion: {q['Question']}\n\tAnswer: {q['Answer']}\n")
        continue
    
    question_str = f"Question: {q['Question']}\n\nOptions:\n{options_str}\n\nAnswer:"
    dataset_json.append({'subject': q['field'], 'question': question_str, 'answer': letter_answer})
    

for q in tqdm(list_float_qs + list_int_qs):

    options_str, letter_answer = generate_random_options_for_list_answers(q['Answer'])

    if options_str is None:
        print(f"Skipping the following question, could not generate options in enough iterations:\n\tQuestion: {q['Question']}\n\tAnswer: {q['Answer']}\n")
        continue

    question_str = f"Question: {q['Question']}\n\nOptions:\n{options_str}\n\nAnswer:"
    dataset_json.append({'subject': q['field'], 'question': question_str, 'answer': letter_answer})

print(f"Writing a total of {len(dataset_json)} MCQs for theorem QA")
write_jsonl(dataset_json, f'../datasets/TheoremQA_mcqa.jsonl')
# -


