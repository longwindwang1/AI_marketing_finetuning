"""
数据准备脚本 - 短视频广告流量分析训练数据

功能：
1. 提供 seed 样本模板，用于指导大模型批量生成训练数据
2. 将生成的原始数据转换为训练格式 (ChatML)
3. 拆分训练集和验证集
"""

import json
import random
import os
from pathlib import Path

# ============================================
# 第一部分：Seed 样本（用于指导数据生成）
# ============================================

SEED_EXAMPLES = [
    # ---- 类别1: 投放数据分析 ----
    {
        "instruction": "请帮我分析以下抖音信息流广告的投放数据，并给出优化建议。\n\n广告计划：夏季连衣裙促销\n投放时间：2024-06-15 至 2024-06-22（7天）\n总消耗：¥15,000\n展示量：500,000\n点击量：8,500\n转化量（下单）：120\n客单价：¥89\n\nCPM：¥30\nCTR：1.7%\nCVR：1.41%\nCPA：¥125\nROI：0.71",
        "output": """## 投放数据分析报告

### 核心指标诊断

| 指标 | 当前值 | 行业基准（服装类） | 状态 |
|------|--------|-------------------|------|
| CPM | ¥30 | ¥20-35 | ✅ 正常 |
| CTR | 1.7% | 2.0%-3.5% | ⚠️ 偏低 |
| CVR | 1.41% | 1.5%-3.0% | ⚠️ 偏低 |
| CPA | ¥125 | ¥60-90 | ❌ 过高 |
| ROI | 0.71 | 1.5+ | ❌ 亏损 |

### 问题诊断

**1. CTR 偏低（1.7% vs 基准 2.0%+）**
- 素材吸引力不足，前3秒未能有效抓住用户注意力
- 可能存在人群定向过宽，展示给了低意向用户

**2. CVR 偏低（1.41% vs 基准 1.5%+）**
- 落地页与素材承诺可能存在断裂
- 价格锚点或促销力度可能不够有吸引力
- 购买决策路径可能过长

**3. ROI 为 0.71，处于亏损状态**
- 营收 = 120 × ¥89 = ¥10,680
- 成本 = ¥15,000
- 亏损 ¥4,320

### 优化建议

**素材优化（提升 CTR）：**
1. 前3秒加入强钩子：试穿对比、价格冲击、场景痛点
2. 增加真人出镜试穿展示，增强代入感
3. 制作 3-5 组不同角度的素材进行 A/B 测试

**落地页优化（提升 CVR）：**
1. 确保落地页首屏与素材内容一致
2. 增加限时优惠倒计时，营造紧迫感
3. 添加用户评价和销量数据作为社会证明

**投放策略调整：**
1. 缩窄人群定向：聚焦 22-35 岁女性，有服装消费行为
2. 分时段投放：重点覆盖 12:00-14:00 和 20:00-23:00 高峰时段
3. 先降低日预算至 ¥1,000，测试优化后素材的效果
4. 目标 CPA 设置为 ¥80，让系统自动优化

**预期效果：** 若 CTR 提升至 2.5%，CVR 提升至 2.0%，CPA 可降至约 ¥60，ROI 有望达到 1.5 以上。""",
        "category": "data_analysis"
    },

    # ---- 类别2: 流量诊断 ----
    {
        "instruction": "我的抖音广告昨天突然不起量了，之前每天能消耗 ¥5000，昨天只跑了 ¥800。没有改过任何设置。请帮我分析可能的原因和解决方案。",
        "output": """## 流量骤降诊断分析

### 可能原因排查（按概率排序）

**1. 素材疲劳 / 创意衰退（概率最高 ⭐⭐⭐）**
- 同一素材跑了一段时间后，目标人群已经大量曝光过
- 系统判定素材 eCPM 下降，减少分配流量
- **判断方法：** 检查近 7 天 CTR 趋势是否持续下降

**2. 竞争环境变化（概率高 ⭐⭐⭐）**
- 同行业竞品加大投放力度（如大促前期）
- 导致 CPM 上涨，相同出价拿不到量
- **判断方法：** 检查 CPM 是否近期有明显上涨

**3. 平台流量波动（概率中 ⭐⭐）**
- 周中 vs 周末流量差异
- 平台算法更新或流量分配策略调整
- **判断方法：** 同行是否也出现类似情况

**4. 账户/计划触发风控（概率中 ⭐⭐）**
- 素材可能被审核重新标记
- 账户信用分变化
- **判断方法：** 检查广告审核状态和账户通知

**5. 人群探索饱和（概率中 ⭐⭐）**
- 定向人群包已被充分探索，系统找不到新的转化用户
- **判断方法：** 检查人群覆盖率和频次数据

### 解决方案

**紧急措施（今天就做）：**
1. 新建 2-3 条计划，复制原计划设置但使用新素材
2. 不要修改原计划（避免重置模型）
3. 准备至少 3 套新素材备用

**短期优化（1-3天内）：**
1. 拓宽人群定向，给系统更多探索空间
2. 适当提高出价 10%-20% 测试是否是竞价问题
3. 尝试不同的投放时段组合

**长期机制：**
1. 建立素材库轮换机制，每周至少更新 2-3 条新素材
2. 同时跑 5-8 条计划，避免依赖单一计划
3. 监控 CTR 连续 3 天下降时，主动更换素材""",
        "category": "traffic_diagnosis"
    },

    # ---- 类别3: 素材分析 ----
    {
        "instruction": "我有一条短视频广告脚本，请帮我分析转化潜力并提出改进建议。\n\n产品：减脂代餐奶昔\n视频时长：15秒\n脚本：\n0-3秒：产品包装特写，文字「天然成分，健康瘦身」\n3-8秒：模特在厨房冲泡产品，展示操作过程\n8-12秒：模特喝完后微笑，竖大拇指\n12-15秒：产品信息 + 价格 + 购买引导",
        "output": """## 素材转化潜力分析

### 整体评分：5/10 ⚠️ 有较大优化空间

### 逐段分析

**0-3秒 - 开头（❌ 最大问题）**
- 产品包装特写 + 文字标语 = 典型「广告感开头」
- 用户在信息流中 1-2 秒内决定是否划走
- 当前开头没有制造冲突、好奇或共鸣，划走率会很高
- **前3秒是决定 CTR 的核心，必须重点优化**

**3-8秒 - 中段（⚠️ 一般）**
- 操作过程展示可以保留，但缺少信息增量
- 没有传达「为什么选这个」的关键卖点
- 纯展示操作不够有说服力

**8-12秒 - 信任构建（❌ 薄弱）**
- 微笑竖大拇指是最弱的信任构建方式
- 缺少数据、对比、用户证言等说服力元素

**12-15秒 - 收尾转化（⚠️ 一般）**
- 常规的产品+价格+引导还可以
- 缺少紧迫感和行动理由

### 优化后脚本建议

```
0-3秒【钩子 - 制造冲突】
画面：体重秤数字特写 + 文字「节食30天，一斤没瘦？」
（或：真人出镜，表情沮丧看着秤）
→ 精准击中目标人群痛点，阻止划走

3-6秒【解决方案引入】
画面：「直到我用了这个方法」→ 拿起产品冲泡
旁白/文字：一杯=一顿饭的营养，只有 XX 卡路里

6-10秒【信任证据】
画面：前后对比图（需合规）/ 成分表特写 / 真实用户反馈截图
文字：XX万人已验证 / 某权威认证

10-13秒【临门一脚】
画面：多口味展示 + 价格锚点
文字：原价¥199，直播间专属¥99 / 前100名再送XX

13-15秒【行动指令】
画面：点击购物车动画引导
文字：「点击下方立即抢购」+ 倒计时
```

### 关键改进点总结
1. **前3秒必须换成痛点钩子**，而非产品特写
2. **增加数据和社会证明**，取代主观的竖大拇指
3. **加入价格锚点和稀缺性**，提升转化紧迫感
4. 整体叙事结构：痛点 → 方案 → 证据 → 行动""",
        "category": "creative_analysis"
    },

    # ---- 类别4: 受众分析 ----
    {
        "instruction": "我们是一个在线教育品牌，主打职场技能课程（Excel、PPT、Python等），准备在抖音投放短视频广告。请帮我分析目标受众画像和定向策略。",
        "output": """## 目标受众分析与定向策略

### 核心受众画像

**第一梯队：职场新人（转化意愿最高）**
- 年龄：22-28岁
- 特征：应届生 / 工作1-3年，面临技能不足的焦虑
- 痛点：工作效率低、被同事碾压、想升职加薪
- 活跃时间：午休(12-14点)、晚间(20-23点)
- 内容偏好：职场干货、逆袭故事、效率工具

**第二梯队：转行/提升人群**
- 年龄：25-35岁
- 特征：对现状不满，想转行或提升竞争力
- 痛点：行业内卷、技能单一、不知如何突破
- 消费能力：中等，对有明确效果的课程愿意付费

**第三梯队：大学生群体**
- 年龄：18-23岁
- 特征：即将就业，想提前储备技能
- 痛点：简历没有亮点、不知学什么
- 注意：付费能力较低，适合引流低价课

### 抖音定向策略

**基础定向：**
- 年龄：22-35岁（核心），18-22岁（拓展）
- 地域：一二线城市为主（职场需求强）
- 性别：不限（但可分别制作素材测试）

**兴趣行为定向：**
- 兴趣关键词：职场提升、办公技巧、副业赚钱、Excel教程、Python学习
- 行为定向：搜索过「Excel教程」「职场技能」等关键词
- 达人粉丝定向：关注秋叶PPT、Excel函数大全等同类账号的用户

**人群包策略：**
1. **种子人群：** 已有付费学员的手机号上传，生成 lookalike 人群
2. **DMP人群：** 教育行业、知识付费兴趣人群
3. **排除人群：** 已转化用户、同行从业者

### 投放节奏建议

**第一周（测试期）：**
- 日预算 ¥500-1000
- 准备 5-8 套素材 × 3 组人群定向
- 跑出 CTR > 3%、CVR > 2% 的优质组合

**第二周（放量期）：**
- 优质计划提升预算至 ¥2000-5000/天
- 基于数据缩窄或拓展定向

**素材方向建议：**
1. 「学会这个 Excel 技巧，同事都以为我开挂了」— 效果展示型
2. 「月薪 3000 到 15000，我只做了一件事」— 逆袭故事型
3. 「你的同事正在偷偷学这个」— 焦虑驱动型""",
        "category": "audience_analysis"
    },

    # ---- 类别5: ROI 优化 ----
    {
        "instruction": "我在快手投放短视频带货广告，目前数据如下：\n\n日预算：¥3,000\n日均消耗：¥2,800\n日均订单：35单\n客单价：¥150\n退货率：25%\n商品成本率：40%\n\n请帮我计算实际ROI，并给出提升ROI的策略。",
        "output": """## ROI 精算与优化策略

### 实际 ROI 计算

**表面数据：**
- 日均 GMV = 35 × ¥150 = ¥5,250
- 表面 ROI = ¥5,250 / ¥2,800 = **1.88**

**扣除退货后：**
- 实际成交单数 = 35 × (1 - 25%) = 26.25 单
- 实际 GMV = 26.25 × ¥150 = ¥3,937.5
- 实际 ROI = ¥3,937.5 / ¥2,800 = **1.41**

**利润维度：**
- 实际营收：¥3,937.5
- 商品成本：¥3,937.5 × 40% = ¥1,575
- 广告成本：¥2,800
- 毛利 = ¥3,937.5 - ¥1,575 - ¥2,800 = **-¥437.5**
- 净 ROI（利润/广告费）= **-15.6%**

### ⚠️ 核心问题：当前处于亏损状态

要实现盈亏平衡，需要满足：
- 实际 GMV > 广告费 / (1 - 成本率) = ¥2,800 / 0.6 = ¥4,667
- 即需要：至少 31 个实际成交单（考虑退货后约 42 个下单）

### ROI 提升策略

**策略一：降低退货率（优先级最高 ⭐⭐⭐）**

退货率从 25% 降到 15%，影响巨大：
- 实际成交 = 35 × 0.85 = 29.75 单
- 实际 GMV = ¥4,462.5
- 毛利 = ¥4,462.5 - ¥1,785 - ¥2,800 = **-¥122.5**（接近盈亏平衡）

具体措施：
1. 素材中如实展示产品，减少预期偏差
2. 加强客服在下单后的确认跟进
3. 分析退货原因 TOP3，针对性解决

**策略二：提升客单价**

客单价从 ¥150 提升到 ¥180（搭配销售 / 满减）：
- 实际 GMV = 26.25 × ¥180 = ¥4,725
- 毛利 = ¥4,725 - ¥1,890 - ¥2,800 = **+¥35**（扭亏）

**策略三：优化广告效率，降低 CPA**

当前 CPA = ¥2,800 / 35 = ¥80/单
目标 CPA < ¥67（盈亏平衡点）

措施：
1. 优化素材提升 CVR，降低单客获取成本
2. 分析成交订单的人群特征，收窄定向
3. 分时段分析 ROI，砍掉低效时段

**策略四：组合拳（推荐）**
- 退货率降至 18% + 客单价提升至 ¥170 + CPA 降至 ¥72
- 预估毛利 = 35 × 0.82 × ¥170 × 0.6 - ¥2,520 = **+¥406/天**
- 月利润约 **¥12,180**

### 监控指标看板建议

| 指标 | 当前 | 目标 | 优先级 |
|------|------|------|--------|
| 退货率 | 25% | <18% | P0 |
| 客单价 | ¥150 | ¥170+ | P1 |
| CPA | ¥80 | <¥72 | P1 |
| 净ROI | -15.6% | >10% | 结果指标 |""",
        "category": "roi_optimization"
    },
]

# 数据生成的 prompt 模板
GENERATION_PROMPT = """你是一个短视频广告流量分析领域的数据生成专家。请基于以下种子样本的风格和质量标准，
生成新的训练样本。

要求：
1. 覆盖以下类别：投放数据分析、流量诊断、素材/创意分析、受众分析、ROI优化
2. 涵盖不同平台：抖音、快手、TikTok、视频号、B站
3. 涵盖不同行业：电商、教育、游戏、本地生活、金融、美妆、食品等
4. 包含具体数字和数据，使分析更真实
5. 回答要专业、结构化、可执行

请生成 {n} 条训练样本，每条包含 instruction 和 output 字段。
输出为 JSON 数组格式。

种子样本参考：
{seed_example}
"""


# ============================================
# 第二部分：数据格式转换
# ============================================

SYSTEM_PROMPT = (
    "你是一位专业的短视频广告流量分析师。你精通各大短视频平台（抖音、快手、TikTok、视频号等）"
    "的广告投放体系，擅长分析广告投放数据、诊断流量问题、优化投放策略、评估素材质量和分析受众特征。"
    "请基于数据和专业经验，给出准确、可执行的分析和建议。"
)


def convert_to_chatml(example: dict) -> dict:
    """将单条样本转换为 ChatML 对话格式"""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def process_raw_data(input_path: str, output_dir: str, eval_ratio: float = 0.1):
    """
    处理原始数据文件，转换格式并拆分训练/验证集

    Args:
        input_path: 原始数据文件路径 (jsonl，每行一个 {instruction, output} 对象)
        output_dir: 输出目录
        eval_ratio: 验证集比例
    """
    examples = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            examples.append(convert_to_chatml(example))

    random.shuffle(examples)
    split_idx = max(1, int(len(examples) * eval_ratio))
    eval_data = examples[:split_idx]
    train_data = examples[split_idx:]

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    eval_path = os.path.join(output_dir, "eval.jsonl")

    for path, data in [(train_path, train_data), (eval_path, eval_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"处理完成！训练集: {len(train_data)} 条, 验证集: {len(eval_data)} 条")
    print(f"训练集路径: {train_path}")
    print(f"验证集路径: {eval_path}")

    return train_path, eval_path


def save_seed_examples():
    """将 seed 样本保存为原始数据文件"""
    raw_dir = Path(__file__).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "seed_examples.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for example in SEED_EXAMPLES:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Seed 样本已保存至: {output_path}")
    print(f"共 {len(SEED_EXAMPLES)} 条种子样本")
    return str(output_path)


def print_generation_prompt(n: int = 10):
    """打印用于大模型生成数据的 prompt"""
    seed = random.choice(SEED_EXAMPLES)
    seed_str = json.dumps(seed, ensure_ascii=False, indent=2)
    prompt = GENERATION_PROMPT.format(n=n, seed_example=seed_str)
    print("=" * 60)
    print("复制以下 Prompt 到 Claude/GPT 中生成训练数据：")
    print("=" * 60)
    print(prompt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="短视频广告流量分析 - 数据准备工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 保存种子样本
    subparsers.add_parser("seed", help="保存种子样本到 raw 目录")

    # 生成 prompt
    gen_parser = subparsers.add_parser("prompt", help="生成用于大模型扩充数据的 prompt")
    gen_parser.add_argument("-n", type=int, default=10, help="每次生成的样本数量")

    # 处理数据
    proc_parser = subparsers.add_parser("process", help="处理原始数据为训练格式")
    proc_parser.add_argument("--input", required=True, help="原始数据文件路径")
    proc_parser.add_argument("--output", default="../data/processed", help="输出目录")
    proc_parser.add_argument("--eval-ratio", type=float, default=0.1, help="验证集比例")

    args = parser.parse_args()

    if args.command == "seed":
        save_seed_examples()
    elif args.command == "prompt":
        print_generation_prompt(args.n)
    elif args.command == "process":
        process_raw_data(args.input, args.output, args.eval_ratio)
    else:
        # 默认行为：保存种子样本并处理
        print("使用方法:")
        print("  1. 保存种子样本:  python prepare_dataset.py seed")
        print("  2. 生成扩充prompt: python prepare_dataset.py prompt -n 20")
        print("  3. 处理原始数据:  python prepare_dataset.py process --input raw/all_data.jsonl")
