import torch
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig, GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from models.model_dpo import AutoDPOModelForCausalLM
import json

# Function to load jsonl files
def load_jsonl(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        return [json.loads(line) for line in file]

def test_predictions():

    # model_path = "ProbLLMs/DPO_Checkpoint"

    # tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False,
    #                                               cache_dir='./cached_models')

    # tokenizer.pad_token_id = 100_000 - 1  # Some random chinese character

    # model = AutoPeftModelForCausalLM.from_pretrained(model_path, cache_dir='./cached_models').to('cuda')
    # model.generation_config = GenerationConfig.from_pretrained(model_path)

    # # Some settings
    # model.config.use_cache = False
    # model.generation_config.pad_token_id = model.generation_config.eos_token_id
    # tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(model.generation_config)

    # Wrap the model with AutoDPOModelForCausalLM
    dpo_model = AutoDPOModelForCausalLM(pretrained_model=model)

    n_samples = 2

    batch = load_jsonl("./datasets/dpo_preference_eval_student_only.jsonl")[:n_samples]

    print(batch)
    
    preference_batch = {}

    preference_batch['prompt'] = [b['prompt'] for b in batch]
    preference_batch['chosen'] = [b['chosen'] for b in batch]
    preference_batch['rejected'] = [b['rejected'] for b in batch]


    chosen_logps_list, rejected_logps_list = dpo_model.get_logprobs(preference_batch, tokenizer)


    print("Chosen logps:", chosen_logps_list)
    print("Rejected logps:", rejected_logps_list)


    # mcqa_batch = {
    #     "question": ["What is the capital of France? A) Paris B) Berlin C) Rome", "What is the capital of Italy? A) Paris B) Berlin C) Rome"],
    #     "answer": ["A"]
    # }

    # mcqa_batch = {}

    # batch = load_jsonl("./datasets/mcqa_example.jsonl")[:n_samples]

    # mcqa_batch['question'] = [b['question'] for b in batch]
    # mcqa_batch['answer'] = [b['answer'] for b in batch]

    # print(mcqa_batch)

    # mcqa_output = dpo_model.prediction_step_mcqa(mcqa_batch, tokenizer)
    # print("Prediction step MCQA output:")
    # print(mcqa_output)

if __name__ == "__main__":
    test_predictions()