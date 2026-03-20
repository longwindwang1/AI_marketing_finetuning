"""
推理脚本 - 加载微调后的模型进行对话推理

支持：
1. 加载 LoRA adapter + 基座模型
2. 合并 adapter 导出完整模型
3. 交互式对话
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


SYSTEM_PROMPT = (
    "你是一位专业的短视频广告流量分析师。你精通各大短视频平台（抖音、快手、TikTok、视频号等）"
    "的广告投放体系，擅长分析广告投放数据、诊断流量问题、优化投放策略、评估素材质量和分析受众特征。"
    "请基于数据和专业经验，给出准确、可执行的分析和建议。"
)


def load_model(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_path: str = None,
    load_in_4bit: bool = True,
):
    """
    加载模型

    Args:
        base_model: 基座模型名称或路径
        adapter_path: LoRA adapter 路径（None 则加载原始模型对比）
        load_in_4bit: 是否 4-bit 量化加载（节省显存）
    """
    print(f"加载基座模型: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    if adapter_path:
        print(f"加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Adapter 加载完成")

    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    query: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """生成回复"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # 只取新生成的 token
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response


def merge_and_export(
    base_model: str,
    adapter_path: str,
    output_path: str,
):
    """合并 LoRA adapter 到基座模型并导出"""
    print("加载基座模型（全精度）...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("加载并合并 adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    print(f"保存合并后的模型至: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("导出完成！")


def interactive_chat(model, tokenizer):
    """交互式对话"""
    print("\n" + "=" * 50)
    print("短视频广告流量分析助手 - 交互模式")
    print("输入问题开始分析，输入 'quit' 退出")
    print("=" * 50 + "\n")

    while True:
        query = input("\n📊 你的问题: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not query:
            continue

        print("\n🤖 分析中...\n")
        response = generate(model, tokenizer, query)
        print(response)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型推理")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="基座模型",
    )
    parser.add_argument(
        "--adapter", type=str, default=None, help="LoRA adapter 路径"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="合并 adapter 并导出",
    )
    parser.add_argument(
        "--merge-output", type=str, default="./merged_model", help="合并模型输出路径"
    )
    parser.add_argument(
        "--no-4bit", action="store_true", help="不使用 4-bit 量化"
    )
    parser.add_argument(
        "--query", type=str, default=None, help="单次查询（不提供则进入交互模式）"
    )

    args = parser.parse_args()

    if args.merge:
        if not args.adapter:
            print("错误：合并模式需要指定 --adapter 路径")
            exit(1)
        merge_and_export(args.base_model, args.adapter, args.merge_output)
    else:
        model, tokenizer = load_model(
            args.base_model,
            args.adapter,
            load_in_4bit=not args.no_4bit,
        )

        if args.query:
            response = generate(model, tokenizer, args.query)
            print(response)
        else:
            interactive_chat(model, tokenizer)
