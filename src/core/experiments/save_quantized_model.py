import hydra
import torch
from accelerate import Accelerator
from transformers import set_seed

from probllms.models import Models, tokenize_and_generate

from huggingface_hub import login

@hydra.main(config_path="./config", config_name="default_generation", version_base="1.2")
def run_experiment(hydra_config):
    model_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
        "load_in_4bit": True,
        "device_map": {"": Accelerator().local_process_index},
    }

    model, tokenizer = Models[hydra_config.model_name_or_path].load_peft_model_and_tokenizer(
        model_tokenizer_path=hydra_config.peft_model_path,
        cache_dir=hydra_config.cache_dir,
        model_config=model_config
    )

    model.save_pretrained(save_directory=f"out/Quantized-{hydra_config.huggingface_name}")
    tokenizer.save_pretrained(f"out/Quantized-{hydra_config.huggingface_name}")
    #model.push_to_hub(f"ProbLLMs/Quantized-{hydra_config.huggingface_name}")

if __name__ == "__main__":
    run_experiment()
