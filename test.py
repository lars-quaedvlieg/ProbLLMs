import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from models.model_dpo import AutoDPOModelForCausalLM 
import json

# Function to load jsonl files
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def test_predictions():

    model_path = "./checkpoints/final_checkpoint/"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Wrap the model with AutoDPOModelForCausalLM
    dpo_model = AutoDPOModelForCausalLM(pretrained_model=model)

    mcqa_batch = {
        "question": [ "What is the capital of France? A) Paris B) Berlin C) Rome"],
        "answer": ["A"]
    }
    
    mcqa_batch = load_jsonl("./datasets/mcqa_example.jsonl")[0]

    mcqa_output = dpo_model.prediction_step_mcqa(mcqa_batch, tokenizer)
    print("Prediction step MCQA output:")
    print(mcqa_output)

if __name__ == "__main__":
    test_predictions()