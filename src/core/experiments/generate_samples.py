import hydra
import torch
from accelerate import Accelerator
from transformers import set_seed

from probllms.models import Models, tokenize_and_generate


@hydra.main(config_path="./config", config_name="default_generation", version_base="1.2")
def run_experiment(hydra_config):
    # Set up logging with HuggingFace
    set_seed(hydra_config.seed)

    # 1. load a pretrained model
    torch_dtype = torch.float
    if hydra_config.model_dtype == "float16":
        torch_dtype = torch.float16
    elif hydra_config.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
        "load_in_4bit": hydra_config.load_in_4bit,
        "device_map": {"": Accelerator().local_process_index},
    }

    model, tokenizer = Models[hydra_config.model_name_or_path].load_peft_model_and_tokenizer(
        model_tokenizer_path=hydra_config.peft_model_path,
        cache_dir=hydra_config.cache_dir,
        model_config=model_config
    )

    ### This is just for testing the model! ###
    gen_config = {
        "max_new_tokens": hydra_config.max_generation_length,
    }
    while True:
        prompt = input("Ask a question: ")
        gen_tokens, output = tokenize_and_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config=gen_config,
        )
        print(type(gen_tokens))
        print(output)
    #################################################

if __name__ == "__main__":
    run_experiment()
