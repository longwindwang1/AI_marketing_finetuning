"""
模型评估脚本

评估维度：
1. 基础指标：loss、perplexity
2. 对比评估：微调前 vs 微调后在相同问题上的回答质量
3. 分类别评估：各任务类别的表现
"""

import json
import torch
from pathlib import Path

# 添加父目录到路径以导入 inference 模块
import sys
sys.path.append(str(Path(__file__).parent.parent))

from inference.inference import load_model, generate


# 评估测试用例
EVAL_CASES = [
    {
        "category": "data_analysis",
        "name": "投放数据分析",
        "query": (
            "请分析以下抖音广告数据：\n"
            "产品：护肤精华液\n"
            "投放3天，消耗 ¥8,000\n"
            "展示量 200,000，点击 4,200，转化（加购）380\n"
            "CTR 2.1%，CVR 9.05%，CPM ¥40，CPA ¥21.05\n"
            "请给出分析和优化建议。"
        ),
        "eval_criteria": [
            "是否对各项指标给出了合理的判断（正常/偏高/偏低）",
            "是否与行业基准进行了对比",
            "是否给出了具体可执行的优化建议",
            "分析是否有结构和层次",
        ],
    },
    {
        "category": "traffic_diagnosis",
        "name": "流量诊断",
        "query": (
            "我的快手广告 CTR 从 3.2% 降到了 1.5%，但 CVR 保持稳定在 2%。"
            "消耗也跟着下降了一半。请帮我诊断问题。"
        ),
        "eval_criteria": [
            "是否正确识别了 CTR 下降但 CVR 稳定的含义",
            "是否排查了素材疲劳、竞争等可能原因",
            "是否给出了针对性的解决方案",
        ],
    },
    {
        "category": "creative_analysis",
        "name": "素材分析",
        "query": (
            "评估这个短视频广告脚本的转化潜力：\n"
            "产品：儿童编程课\n"
            "0-3秒：可爱的小朋友在电脑前编程\n"
            "3-8秒：展示孩子做出的小游戏\n"
            "8-12秒：家长testimonial说效果好\n"
            "12-15秒：课程介绍 + 限时优惠"
        ),
        "eval_criteria": [
            "是否逐段分析了脚本的优劣",
            "是否指出了前3秒的钩子设计问题",
            "是否给出了具体的改进脚本建议",
        ],
    },
    {
        "category": "audience_analysis",
        "name": "受众分析",
        "query": "我们是一个宠物食品品牌，主打高端天然猫粮，准备在抖音投放。请分析目标受众和定向策略。",
        "eval_criteria": [
            "是否描述了详细的人群画像",
            "是否给出了具体的定向策略",
            "是否考虑了兴趣行为和达人定向",
        ],
    },
    {
        "category": "roi_optimization",
        "name": "ROI优化",
        "query": (
            "视频号广告数据：日消耗 ¥5000，GMV ¥8000，ROI 1.6，"
            "退货率 30%，商品成本率 45%。请计算实际利润并给优化建议。"
        ),
        "eval_criteria": [
            "是否正确计算了扣除退货后的实际ROI",
            "是否计算了利润/亏损金额",
            "是否给出了多维度的ROI优化建议",
        ],
    },
]


def run_evaluation(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_path: str = None,
    output_path: str = None,
):
    """
    运行评估

    Args:
        base_model: 基座模型
        adapter_path: 微调后的 adapter 路径
        output_path: 评估结果输出路径
    """
    model, tokenizer = load_model(base_model, adapter_path)
    model_label = "finetuned" if adapter_path else "base"

    results = []
    print(f"\n{'='*60}")
    print(f"评估模型: {model_label}")
    print(f"{'='*60}\n")

    for i, case in enumerate(EVAL_CASES):
        print(f"\n[{i+1}/{len(EVAL_CASES)}] {case['name']} ({case['category']})")
        print("-" * 40)

        response = generate(model, tokenizer, case["query"])

        result = {
            "category": case["category"],
            "name": case["name"],
            "query": case["query"],
            "response": response,
            "eval_criteria": case["eval_criteria"],
            "model": model_label,
        }
        results.append(result)

        # 打印回答前 500 字符作为预览
        preview = response[:500] + "..." if len(response) > 500 else response
        print(f"\n回答预览:\n{preview}\n")

    # 保存结果
    if output_path is None:
        output_path = f"eval_results_{model_label}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n评估结果已保存至: {output_path}")
    print(f"共评估 {len(results)} 个用例")

    return results


def compare_results(base_results_path: str, finetuned_results_path: str):
    """对比基座模型和微调模型的评估结果"""

    def load_results(path):
        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        return results

    base_results = load_results(base_results_path)
    ft_results = load_results(finetuned_results_path)

    print("\n" + "=" * 70)
    print("基座模型 vs 微调模型 对比评估")
    print("=" * 70)

    for base_r, ft_r in zip(base_results, ft_results):
        print(f"\n{'─'*70}")
        print(f"📋 类别: {base_r['name']} ({base_r['category']})")
        print(f"📝 问题: {base_r['query'][:80]}...")
        print(f"\n🔵 基座模型回答长度: {len(base_r['response'])} 字符")
        print(f"🟢 微调模型回答长度: {len(ft_r['response'])} 字符")
        print(f"\n评估标准:")
        for criterion in base_r["eval_criteria"]:
            print(f"  - {criterion}")

    print(f"\n{'='*70}")
    print("提示：请人工对照评估标准，判断两个模型的回答质量差异")
    print("也可以将结果文件提供给 Claude/GPT 进行自动评分")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter 路径")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASE_RESULTS", "FT_RESULTS"),
        help="对比两个评估结果文件",
    )

    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        run_evaluation(args.base_model, args.adapter, args.output)
