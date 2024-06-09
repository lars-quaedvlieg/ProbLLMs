from enum import Enum
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig 

class Models(Enum):
    DeepSeekMath_7B = "deepseek-ai/deepseek-math-7b-rl"
    Llama3_8B = "meta-llama/Meta-Llama-3-8B"
    Llama3_70B = "meta-llama/Meta-Llama-3-70B"
    GPT2 = "openai-community/gpt2"
    Phi3 = "microsoft/Phi-3-mini-4k-instruct"

    def load_model_and_tokenizer(self, cache_dir: str, model_config: dict[str, Any], quantize: bool = False):
        model_name = self.value
        # add_eos_token=True is very important here, since we do fine-tuning
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if quantize:
            # Quantization config for 8 bit
            quantization_config = BitsAndBytesConfig(
                load_in_8bit = True,
            )
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_config, cache_dir=cache_dir, quantization_config=quantization_config)
            model.generation_config = GenerationConfig.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_config, cache_dir=cache_dir)
            model.generation_config = GenerationConfig.from_pretrained(model_name)

        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            print('Added custom padding token')
            tokenizer.add_special_tokens({"pad_token": "<<|[PAD]|>>"})
            model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def load_peft_model_and_tokenizer(self, model_tokenizer_path: str, cache_dir: str, model_config: dict[str, Any]):
        model_name = self.value
        tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path, cache_dir=cache_dir)

        model = AutoPeftModelForCausalLM.from_pretrained(model_tokenizer_path, **model_config, cache_dir=cache_dir)
        model.generation_config = GenerationConfig.from_pretrained(model_name)

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