from enum import Enum
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class Models(Enum):
    DeepSeekMath_7B = "deepseek-ai/deepseek-math-7b-rl"
    Llama3_8B = "meta-llama/Meta-Llama-3-8B"
    Llama3_70B = "meta-llama/Meta-Llama-3-70B"

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
