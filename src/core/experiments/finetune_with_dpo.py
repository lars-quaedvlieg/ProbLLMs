import os

import hydra
import torch
from accelerate import Accelerator
from peft import LoraConfig
from transformers import set_seed, TrainingArguments
from trl import DPOTrainer

from probllms.data import get_dataset
from probllms.models import Models


@hydra.main(config_path="./config", config_name="default_dpo_lora", version_base="1.2")
def run_experiment(hydra_config):
    # Set up logging with HuggingFace
    if hydra_config.wandb.active:
        os.environ["WANDB_PROJECT"] = hydra_config.wandb.project
        os.environ["WANDB_ENTITY"] = hydra_config.wandb.entity
        os.environ["WANDB_NAME"] = hydra_config.wandb.name
        os.environ["WANDB_DIR"] = hydra_config.wandb.log_directory

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

    model, tokenizer = Models[hydra_config.model_name_or_path].load_model_and_tokenizer(
        cache_dir=hydra_config.cache_dir,
        model_config=model_config
    )

    # ### TODO: This is just for testing the model! ###
    # gen_config = {
    #     "max_new_tokens": 1024,
    # }
    # while True:
    #     prompt = input("Ask a question: ")
    #     gen_tokens, output = tokenize_and_generate(
    #         model=model,
    #         tokenizer=tokenizer,
    #         prompt=prompt,
    #         generation_config=gen_config,
    #     )
    #     print(type(gen_tokens))
    #     print(output)
    # #################################################

    # Load the training dataset
    train_dataset = get_dataset(file_path=hydra_config.train_data_path)
    # Filter questions that are too long
    initial_len = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= hydra_config.max_length
                  and len(x["prompt"]) + len(x["rejected"]) <= hydra_config.max_length
    )
    print(f'Kept {round(len(train_dataset) / initial_len, 2)*100:<.2f}% of the training data')

    # Load the evaluation dataset
    eval_dataset = get_dataset(file_path=hydra_config.eval_data_path)
    # Filter questions that are too long
    initial_len = len(eval_dataset)
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= hydra_config.max_length
                  and len(x["prompt"]) + len(x["rejected"]) <= hydra_config.max_length
    )
    print(f'Kept {round(len(eval_dataset) / initial_len, 2)*100:<.2f}% of the evaluation data')

    # initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=hydra_config.per_device_train_batch_size,
        per_device_eval_batch_size=hydra_config.per_device_eval_batch_size,
        max_steps=hydra_config.max_steps,
        logging_dir=hydra_config.output_dir,
        logging_steps=hydra_config.logging_steps,
        save_steps=hydra_config.save_steps,
        gradient_accumulation_steps=hydra_config.gradient_accumulation_steps,
        gradient_checkpointing=hydra_config.gradient_checkpointing,
        learning_rate=hydra_config.learning_rate,
        evaluation_strategy="steps",
        eval_steps=hydra_config.eval_steps,
        output_dir=hydra_config.output_dir,
        report_to="wandb" if hydra_config.wandb.active else None,
        lr_scheduler_type=hydra_config.lr_scheduler_type,
        warmup_steps=hydra_config.warmup_steps,
        optim=hydra_config.optimizer_type,
        remove_unused_columns=False,
        run_name=hydra_config.wandb.name,
        gradient_checkpointing_kwargs=dict(use_reentrant=hydra_config.gradient_checkpointing_use_reentrant),
        seed=hydra_config.seed,
    )

    peft_config = LoraConfig(
        r=hydra_config.lora_r,
        lora_alpha=hydra_config.lora_alpha,
        lora_dropout=hydra_config.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,  # Will automatically create a reference model from the original
        args=training_args,
        beta=hydra_config.beta,
        train_dataset=train_dataset["train"],
        eval_dataset=eval_dataset["train"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=hydra_config.max_prompt_length,
        max_length=hydra_config.max_length,
    )

    print(model)

    # 6. train
    dpo_trainer.train()

    # 7. save
    output_dir = os.path.join(hydra_config.output_dir, hydra_config.wandb.name)
    dpo_trainer.save_model(output_dir)
    dpo_trainer.model.save_pretrained(os.path.join(output_dir, "final_checkpoint"))


if __name__ == "__main__":
    run_experiment()
