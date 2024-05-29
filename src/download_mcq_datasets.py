from datasets import load_dataset
import pandas as pd
import json

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


# +
def load_mmlu_subject(subject):

    dataset = load_dataset("cais/mmlu", subject)
    splits = list(dataset.keys())

    dfs = []
    for split in splits:
        dfs.append(pd.DataFrame(dataset[split]))

    return pd.concat(dfs, ignore_index=True)


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

with open('../datasets/mmlu-science-related_mcqa.jsonl', 'w') as f:
    for d in mmlu_json:
        json.dump(d, f)
        f.write('\n')
# -
