# AI Marketing Fine-tuning — Short-Video Ad Traffic Analyst

A QLoRA fine-tuning pipeline that adapts **Qwen2.5-7B-Instruct** into a domain expert for short-video advertising analytics (Douyin / Kuaishou / TikTok / WeChat Channels). The fine-tuned model performs ad-data diagnosis, traffic-drop root-cause analysis, creative scoring, audience targeting, and ROI optimization — tasks where the base instruction model produces generic, structurally inconsistent answers.

---

## Why this project

Short-video advertising operators rely on five recurring analytical tasks: **data diagnosis, traffic troubleshooting, creative scoring, audience profiling, and ROI accounting**. General-purpose LLMs answer these too abstractly and miss platform-specific benchmarks (CTR / CVR / CPM / CPA ranges, ROI break-even math, retention-vs-refund accounting). This project distills that domain knowledge into a 7B model that runs on a single consumer GPU after merging.

---

## Highlights

- **End-to-end pipeline**: dataset construction → 4-bit QLoRA training → evaluation harness → inference / adapter merging
- **Reproducible config-driven training** (`train/config.yaml`) — all hyperparameters in one file, no hidden defaults
- **Side-by-side eval framework** comparing base vs. fine-tuned model on 5 fixed test cases (one per task category)
- **Memory-efficient**: 4-bit NF4 + double quantization + paged 8-bit AdamW + gradient checkpointing — trains on a single 40 GB A100, infers on 12 GB consumer GPUs

---

## Tech Stack

| Layer | Tools |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuning | `PEFT` (LoRA), `bitsandbytes` (4-bit NF4) |
| Training framework | `TRL` (`SFTTrainer`), `Transformers`, `Datasets` |
| Optimizer / precision | `paged_adamw_8bit`, `bfloat16`, gradient checkpointing |
| Serving | `Transformers` + `PEFT` adapter loading; optional adapter merge for standalone deployment |
| Language | Python 3.10+, PyTorch 2.x |

---

## Repository Layout

```
ai-marketing-finetune/
├── data/
│   ├── prepare_dataset.py         # seed examples + ChatML conversion + train/eval split
│   ├── raw/                       # generated raw samples (instruction/output pairs)
│   ├── processed/
│   │   ├── train.jsonl            # 176 ChatML conversations
│   │   └── eval.jsonl             # 19 ChatML conversations
│   ├── *_analysis.json            # per-category seed expansions
│   └── generation_prompts.md      # prompts used to bootstrap data via larger LLMs
├── train/
│   ├── config.yaml                # all training hyperparameters
│   ├── train_qlora.py             # SFT training entrypoint
│   └── output/final/              # final LoRA adapter (~323 MB on disk)
├── eval/
│   └── evaluate.py                # base-vs-finetuned comparison on 5 held-out cases
├── inference/
│   └── inference.py               # interactive chat + adapter merge/export
└── requirements.txt
```

---

## Dataset

Domain-specific instruction dataset covering five task categories, generated via seed expansion (5 hand-written gold-quality seeds → expanded with a frontier LLM under category/platform/industry constraints, then human-reviewed):

| Category | Description |
|---|---|
| `data_analysis` | Diagnosing CTR / CVR / CPM / CPA / ROI against industry benchmarks |
| `traffic_diagnosis` | Root-cause analysis for sudden traffic drops or volume changes |
| `creative_analysis` | Per-second short-video script scoring and rewrite suggestions |
| `audience_analysis` | Persona breakdown + targeting / DMP / lookalike strategy |
| `roi_optimization` | True-ROI accounting (refunds, COGS) and lever-by-lever optimization |

- **Format**: ChatML (`system` / `user` / `assistant`) — see `data/prepare_dataset.py`
- **Size**: 176 train / 19 eval (90/10 split, shuffled)
- **Platforms covered**: Douyin, Kuaishou, TikTok, WeChat Channels, Bilibili
- **Industries covered**: e-commerce, education, gaming, local services, finance, beauty, F&B

---

## Training Configuration

```yaml
# QLoRA
quantization:    4-bit NF4 + double quantization (bnb_4bit)
compute dtype:   bfloat16
LoRA rank:       64
LoRA alpha:      128         # 2 × rank
LoRA dropout:    0.05
target modules:  q_proj, k_proj, v_proj, o_proj,
                 gate_proj, up_proj, down_proj   # full attention + MLP

# Optimization
epochs:          3
per-device bs:   4
grad accum:      4           # effective batch = 16
learning rate:   2e-4 (cosine schedule, 5% warmup)
max grad norm:   1.0
max seq length:  2048
optimizer:       paged_adamw_8bit
gradient ckpt:   enabled
precision:       bf16
```

- **Trainable parameters**: ~161 M (≈ 2.3 % of base model)
- **Adapter size on disk**: ~323 MB
- **Trained on**: NVIDIA A100 40 GB (cloud)
- **Inference target**: NVIDIA RTX 5070 12 GB (4-bit loaded base + adapter)

---

## Results

The evaluation harness (`eval/evaluate.py`) runs the base model and the fine-tuned model on five held-out prompts (one per category) with shared system prompt and decoding settings (`temperature=0.7, top_p=0.9, max_new_tokens=2048`). Results are scored against per-case rubrics covering structural quality, benchmark grounding, and actionability.

**Qualitative findings on the five eval cases:**

| Aspect | Base Qwen2.5-7B-Instruct | Fine-tuned (this repo) |
|---|---|---|
| Output structure | Free-form prose, inconsistent | Consistent 3-section format: diagnosis → reasoning → actions |
| Benchmark grounding | Rarely cites industry CTR/CVR ranges | Quotes platform/industry benchmarks per metric |
| ROI accounting | Treats GMV as profit; ignores refunds & COGS | Computes true ROI net of refund rate and cost-of-goods |
| Creative scoring | Generic praise / vague feedback | Per-second timeline analysis + rewritten script |
| Length control | Variable, often over-short | Consistently structured ~600–1200 token answers |

Reproduce with:

```bash
# 1. Run base
python eval/evaluate.py --output eval_results_base.jsonl

# 2. Run fine-tuned
python eval/evaluate.py --adapter train/output/final --output eval_results_finetuned.jsonl

# 3. Side-by-side
python eval/evaluate.py --compare eval_results_base.jsonl eval_results_finetuned.jsonl
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/longwindwang1/AI_marketing_finetuning.git
cd AI_marketing_finetuning
pip install -r requirements.txt
```

### 2. Prepare data

```bash
# write seed examples
python data/prepare_dataset.py seed

# print prompt to expand dataset via Claude / GPT
python data/prepare_dataset.py prompt -n 20

# convert raw jsonl → ChatML and split train/eval
python data/prepare_dataset.py process --input data/raw/all_data.jsonl
```

### 3. Train

```bash
python train/train_qlora.py --config train/config.yaml
```

### 4. Inference

Interactive chat with the fine-tuned adapter:

```bash
python inference/inference.py --adapter train/output/final
```

One-shot query:

```bash
python inference/inference.py \
    --adapter train/output/final \
    --query "我的抖音广告 CTR 从 3% 降到 1.2%，CVR 稳定，请帮我诊断问题。"
```

Merge adapter into the base model for standalone deployment:

```bash
python inference/inference.py \
    --adapter train/output/final \
    --merge \
    --merge-output ./merged_model
```

---

## Hardware Notes

| Stage | GPU | VRAM peak |
|---|---|---|
| Training | A100 40 GB | ~28 GB (bs=4, seq=2048, 4-bit base + bf16 LoRA) |
| Inference (4-bit) | RTX 5070 12 GB | ~7 GB |
| Adapter merge | CPU only | RAM-bound (~30 GB system RAM) |

---

## Future Work

- Replace fixed eval rubric with **LLM-as-judge** scoring (Claude / GPT-4) for automated regression tests
- Expand training set to **1,000+** samples with multi-turn conversations
- Add **retrieval over real ad platform docs** to reduce hallucinated benchmarks
- Explore **DPO / ORPO** on operator-preference pairs once enough feedback is collected
- Quantize the merged model to **GGUF (Q4_K_M)** for CPU/edge deployment via `llama.cpp`

---

## License

MIT.
