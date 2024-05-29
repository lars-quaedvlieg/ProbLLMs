from datasets import load_dataset
import pandas as pd
import json


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
