import json
import os
import random

import jsonlines
from tqdm import tqdm

DATA_PATH = "../datasets"
PREF_FILE_NAME = "M1_preference_data_15052024.json"

ADDITIONAL_DATASET_NAMES = ['camel-ai-math.jsonl', 'camel-ai-physics.jsonl', 'cs_theoryQA.jsonl',
                            'pmp-stack-exchange-1.jsonl', 'pmp-stack-exchange-2.jsonl']

TEST_SET_PROB = 0.2

random.seed(0)

if __name__ == '__main__':
    with open(os.path.join(DATA_PATH, PREF_FILE_NAME), 'r') as f:
        preference_data = json.load(f)

    formatted_train_preference_data = []
    formatted_eval_preference_data = []

    num_wrong_formatted_preferences = 0
    for question_pref_data in tqdm(preference_data):
        num_pairs = len(question_pref_data["preference"])
        for answer_pref_data in question_pref_data["preference"]:
            # Skip answers without a preference
            chosen_answer_char = answer_pref_data["overall"]
            if chosen_answer_char not in {'A', 'B'} or answer_pref_data['criteria']['correctness'] not in {'A', 'B', 'AB'}:
                num_wrong_formatted_preferences += 1
                continue
            rejected_answer_char = 'B' if answer_pref_data["overall"] == 'A' else 'A'

            formatted_pair = {
                "prompt": question_pref_data["question_complete"],
                "chosen": answer_pref_data[chosen_answer_char],
                "rejected": answer_pref_data[rejected_answer_char],
            }

            # We sample uniformly random
            if random.random() < TEST_SET_PROB:
                formatted_eval_preference_data.append(formatted_pair)
            else:
                formatted_train_preference_data.append(formatted_pair)

    print('Student preference pairs:')
    print(f'\tThere are {len(formatted_train_preference_data)} preference pairs in the train dataset!')
    print(f'\tThere are {len(formatted_eval_preference_data)} preference pairs in the evaluation dataset!')
    print(f'\tIn total there were {num_wrong_formatted_preferences} wrong answers skipped preferences...')

    # Now we load our additional datasets
    for dataset_name in ADDITIONAL_DATASET_NAMES:
        with jsonlines.open(os.path.join(DATA_PATH, dataset_name)) as reader:
            for preference_pair in reader:
                # We always want this preference data in the training set
                formatted_train_preference_data.append(preference_pair)

    print('Final preference pairs:')
    print(f'\tThere are {len(formatted_train_preference_data)} preference pairs in the train dataset!')
    print(f'\tThere are {len(formatted_eval_preference_data)} preference pairs in the evaluation dataset!')

    # Writing the train preference dataset
    with jsonlines.open(os.path.join(DATA_PATH, "dpo_preference_train.jsonl"), mode='w') as writer:
        for preference_sample in formatted_train_preference_data:
            writer.write(preference_sample)

    # Writing the eval preference dataset
    with jsonlines.open(os.path.join(DATA_PATH, "dpo_preference_eval.jsonl"), mode='w') as writer:
        for preference_sample in formatted_eval_preference_data:
            writer.write(preference_sample)
