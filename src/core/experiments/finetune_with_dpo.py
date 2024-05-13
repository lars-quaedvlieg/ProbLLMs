import os

import hydra
import torch
import wandb
from accelerate import Accelerator
from omegaconf import omegaconf
from peft import LoraConfig
from transformers import set_seed
from trl import DPOConfig, DPOTrainer

from src.probllms.data import get_dataset
from src.probllms.models import Models


@hydra.main(config_path="./configs", config_name="default_dpo_lora", version_base="1.2")
def run_experiment(hydra_config):
    # Set up logging
    if hydra_config.wandb.active:
        # Parse configuration into dict
        config = omegaconf.OmegaConf.to_container(
            hydra_config, resolve=True, throw_on_missing=True
        )

        wandb.init(
            entity=hydra_config.wandb.entity,
            project=hydra_config.wandb.project,
            name=hydra_config.wandb.name,
            dir=hydra_config.wandb.log_directory,
            config=config,
        )
    else:
        wandb.init(mode="offline", dir=hydra_config.wandb.log_directory)

    set_seed(hydra_config.seed)

    # 1. load a pretrained model
    torch_dtype = torch.float
    if hydra_config.model_dtype == "float16":
        torch_dtype = torch.float16
    elif hydra_config.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model_config = {
        "pretrained_model_name_or_path": hydra_config.model_name_or_path,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
        "load_in_4bit": hydra_config.load_in_4bit,
        "device_map": {"": Accelerator().local_process_index},
    }

    model, tokenizer = Models[hydra_config.model_name_or_path].load_model_and_tokenizer(model_config=model_config)

    # TODO: Load the training dataset
    train_dataset = None
    # train_dataset = get_dataset(data_dir=hydra_config.data_dir)
    # # Filter prompts that are too long
    # train_dataset = train_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= hydra_config.max_length
    #               and len(x["prompt"]) + len(x["rejected"]) <= hydra_config.max_length
    # )

    # TODO: Load the evaluation dataset
    eval_dataset = None
    # eval_dataset = get_dataset(data_dir=hydra_config.eval_data_dir)
    # eval_dataset = eval_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= hydra_config.max_length
    #               and len(x["prompt"]) + len(x["rejected"]) <= hydra_config.max_length
    # )

    # initialize training arguments:
    training_args = DPOConfig(
        per_device_train_batch_size=hydra_config.per_device_train_batch_size,
        per_device_eval_batch_size=hydra_config.per_device_eval_batch_size,
        max_steps=hydra_config.max_steps,
        logging_steps=hydra_config.logging_steps,
        save_steps=hydra_config.save_steps,
        gradient_accumulation_steps=hydra_config.gradient_accumulation_steps,
        gradient_checkpointing=hydra_config.gradient_checkpointing,
        learning_rate=hydra_config.learning_rate,
        evaluation_strategy="steps",
        eval_steps=hydra_config.eval_steps,
        output_dir=hydra_config.output_dir,
        report_to=hydra_config.report_to,
        lr_scheduler_type=hydra_config.lr_scheduler_type,
        warmup_steps=hydra_config.warmup_steps,
        optim=hydra_config.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_llama2",
        gradient_checkpointing_kwargs=dict(use_reentrant=hydra_config.gradient_checkpointing_use_reentrant),
        seed=hydra_config.seed,
    )

    peft_config = LoraConfig(
        r=hydra_config.lora_r,
        lora_alpha=hydra_config.lora_alpha,
        lora_dropout=hydra_config.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=hydra_config.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=hydra_config.max_prompt_length,
        max_length=hydra_config.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(hydra_config.output_dir)

    # 7. save
    output_dir = os.path.join(hydra_config.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)


if __name__ == "__main__":
    run_experiment()
