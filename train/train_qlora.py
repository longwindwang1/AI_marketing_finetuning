"""
QLoRA 微调训练脚本 - 短视频广告流量分析模型

基于 Qwen2.5-7B-Instruct + QLoRA (4-bit NF4)
适配 RTX 5070 (12GB VRAM)
"""

import os
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def load_config(config_path: str = None) -> dict:
    """加载训练配置"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_quantization(config: dict) -> BitsAndBytesConfig:
    """配置 4-bit 量化"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # 嵌套量化，进一步节省显存
    )


def load_model_and_tokenizer(config: dict, bnb_config: BitsAndBytesConfig):
    """加载模型和 tokenizer"""
    model_name = config["model"]["base_model"]
    print(f"正在加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,          # 减少 CPU 内存峰值
    )

    model.config.use_cache = False  # 训练时关闭 KV cache
    model = prepare_model_for_kbit_training(model)

    print(f"模型加载完成，参数量: {model.num_parameters():,}")
    return model, tokenizer


def setup_lora(config: dict) -> LoraConfig:
    """配置 LoRA"""
    qlora_config = config["qlora"]
    return LoraConfig(
        r=qlora_config["r"],
        lora_alpha=qlora_config["lora_alpha"],
        lora_dropout=qlora_config["lora_dropout"],
        target_modules=qlora_config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_data(config: dict):
    """加载训练和验证数据"""
    data_config = config["data"]

    train_path = data_config["train_file"]
    eval_path = data_config["eval_file"]

    # 转为绝对路径（相对于 config 文件位置）
    base_dir = Path(__file__).parent
    train_path = str((base_dir / train_path).resolve())
    eval_path = str((base_dir / eval_path).resolve())

    print(f"训练数据: {train_path}")
    print(f"验证数据: {eval_path}")

    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "eval": eval_path},
    )

    print(f"训练集: {len(dataset['train'])} 条")
    print(f"验证集: {len(dataset['eval'])} 条")

    return dataset


def formatting_func(example):
    """将 messages 格式化为模型输入文本"""
    messages = example["messages"]
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return text


def train(config_path: str = None):
    """主训练流程"""
    config = load_config(config_path)
    train_config = config["training"]

    # 1. 量化配置
    bnb_config = setup_quantization(config)

    # 2. 加载模型
    model, tokenizer = load_model_and_tokenizer(config, bnb_config)

    # 3. LoRA 配置
    lora_config = setup_lora(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 加载数据
    dataset = load_data(config)

    # 5. 训练参数
    output_dir = train_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        warmup_ratio=train_config["warmup_ratio"],
        max_grad_norm=train_config["max_grad_norm"],
        max_length=train_config["max_seq_length"],
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        eval_strategy=train_config["eval_strategy"],
        eval_steps=train_config["eval_steps"],
        bf16=train_config["bf16"],
        gradient_checkpointing=train_config["gradient_checkpointing"],
        optim=train_config["optim"],
        dataloader_num_workers=train_config["dataloader_num_workers"],
        report_to=train_config["report_to"],
        remove_unused_columns=False,
    )

    # 6. 创建 Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # 7. 开始训练
    print("=" * 50)
    print("开始训练...")
    print(f"Epochs: {train_config['num_train_epochs']}")
    print(f"Batch size: {train_config['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {train_config['gradient_accumulation_steps']}")
    effective_batch = (
        train_config["per_device_train_batch_size"]
        * train_config["gradient_accumulation_steps"]
    )
    print(f"Effective batch size: {effective_batch}")
    print(f"Learning rate: {train_config['learning_rate']}")
    print("=" * 50)

    trainer.train()

    # 8. 保存最终模型
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n训练完成！模型已保存至: {final_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QLoRA 微调训练")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    train(args.config)
