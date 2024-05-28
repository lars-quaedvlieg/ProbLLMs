from enum import Enum
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class Models(Enum):
    DeepSeekMath_7B = "deepseek-ai/deepseek-math-7b-rl"
    Llama3_8B = "meta-llama/Meta-Llama-3-8B"
    Llama3_70B = "meta-llama/Meta-Llama-3-70B"
    GPT2 = "openai-community/gpt2"

    def load_model_and_tokenizer(self, cache_dir: str, model_config: dict[str, Any]):
        model_name = self.value
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_config, cache_dir=cache_dir)
        model.generation_config = GenerationConfig.from_pretrained(model_name)

        # Some settings
        model.config.use_cache = False
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def load_peft_model_and_tokenizer(self, model_tokenizer_path: str, cache_dir: str, model_config: dict[str, Any]):
        model_name = self.value
        tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path, cache_dir=cache_dir)
        model = AutoPeftModelForCausalLM.from_pretrained(model_tokenizer_path, **model_config, cache_dir=cache_dir)
        model.generation_config = GenerationConfig.from_pretrained(model_name)

        # Some settings
        model.config.use_cache = False
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer


def tokenize_and_generate(
        model,
        tokenizer,
        prompt: str,
        generation_config: dict[str, Any] = None
) -> (torch.Tensor, str):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), **generation_config)

    generated_tokens = outputs[0][input_tensor.shape[1]:]
    result_str = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_tokens, result_str


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
