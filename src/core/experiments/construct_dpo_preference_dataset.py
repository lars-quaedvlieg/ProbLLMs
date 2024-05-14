import json
import os
import random

import jsonlines
from tqdm import tqdm

DATA_PATH = "../datasets"
PREF_FILE_NAME = "M1_preference_data_07052024.json"

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
            if chosen_answer_char not in {'A', 'B'}:
                num_wrong_formatted_preferences += 1
                continue
            rejected_answer_char = 'B' if answer_pref_data["overall"] == 'A' else 'A'

            formatted_pair = {
                "prompt": question_pref_data["question_complete"],
                "chosen": answer_pref_data[chosen_answer_char],
                "rejected": answer_pref_data[rejected_answer_char],
            }

            # We sample according to the number of question pairs per question, aiming to have one eval per question
            # We offset it by 3, so questions with only 1 pair have a 2/4 chance to get sampled, 2 pairs: 2/5, etc...
            if random.random() < 2. / (num_pairs + 4):
                formatted_eval_preference_data.append(formatted_pair)
            else:
                formatted_train_preference_data.append(formatted_pair)

    print(f'There are {len(formatted_train_preference_data)} preference pairs in the train dataset!')
    print(f'There are {len(formatted_eval_preference_data)} preference pairs in the evaluation dataset!')
    print(f'In total there were {num_wrong_formatted_preferences} incorrectly formatted skipped preferences...')

    # Writing the train preference dataset
    with jsonlines.open(os.path.join(DATA_PATH, "dpo_preference_train.jsonl"), mode='w') as writer:
        for preference_sample in formatted_train_preference_data:
            writer.write(preference_sample)

    # Writing the eval preference dataset
    with jsonlines.open(os.path.join(DATA_PATH, "dpo_preference_eval.jsonl"), mode='w') as writer:
        for preference_sample in formatted_eval_preference_data:
            writer.write(preference_sample)
