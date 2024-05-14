from typing import Dict, Callable

from datasets import Dataset, load_dataset

DEFAULT_FORMAT_FN = lambda question_prompt: question_prompt + "\n\nAnswer: "

### EXAMPLE FILE
def get_dataset(
        file_path: str,
        num_proc=1,
        prompt_format_fn: Callable = DEFAULT_FORMAT_FN,
) -> Dataset:
    dataset = load_dataset(
        "json",
        data_files=file_path,
    )

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": [prompt_format_fn(question) for question in samples["prompt"]],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
    )


if __name__ == '__main__':
    res = get_dataset(
        file_path="../datasets/dpo_preference_train.jsonl"
    )
    print()
