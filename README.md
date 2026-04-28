# DeepCiteFact

A GRPO-based reinforcement learning training framework for long-form factual grounding, simultaneously optimizing fact accuracy, citation compliance, and format integrity through a four-dimensional reward function.

**Key Results:**
- Citation Precision: 0.189 (SFT) → 0.712 (GRPO), 
- Avg Correct Citations: 0.765 → 2.9, 
- Fact Score: 0.719 → 0.785

---

## Project Structure

```
DeepCiteFact/
├── README.md
├── requirements.txt
├── autodl_deploy_guide.txt    # AutoDL deployment guide
├── run_api_version.sh         # API server startup script
├── LlamaFactory/               # SFT training (Qwen3-8B)
│   └── train/sft.yaml
├── data/                       # Training datasets
│   ├── sft_trace_filter.jsonl           # Filtered SFT trajectories (2,918)
│   └── rl_train_data_filter_grpo.jsonl  # GRPO hard samples (1,776)
├── eval/                       # Evaluation scripts (citation + fact metrics)
│   ├── citation_eval.py
│   ├── fact_eval.py
│   └── get_response.py
├── verl/                       # verl framework + custom reward manager
│   ├── verl/workers/reward_manager/
│   │   └── custom.py           # CustomRewardManager (4 reward functions)
│   ├── verl/examples/sglang_multiturn/  # GRPO training config
│   │   └── config/
│   │       ├── search_grpo.yaml
│   │       └── tool_config/custom_tool_config.yaml
│   ├── scripts/run_grpo.sh    # GRPO training launch script
│   └── tensorboard_log/        # Training visualization logs
└── to_hf/                      # Checkpoint export tools
    ├── model_merge.sh
    └── legacy_model_merger.py
```

---

## Four-Dimensional Reward Function

```
R_total = 0.5 × R_fact + 0.35 × R_cite + 0.1 × R_search + 0.05 × R_format
```

| Component | Method | Weight |
|-----------|--------|--------|
| `R_fact` | Qwen2.5-32B atomic claim verification | 0.50 |
| `R_cite` | URL authenticity (0.4) + semantic F1 (0.6) | 0.35 |
| `R_search` | Valid search count / 5 | 0.10 |
| `R_format` | Tag closure validation | 0.05 |

---

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
cd verl && pip install -e . && cd ..
pip install "sglang[all]==0.4.9.post6"
```

### 2. SFT Training

```bash
cd LlamaFactory
llamafactory-cli train train/sft.yaml
```

### 3. Start Reward Judge Server

```bash
# Qwen2.5-32B as reward computation service (vllm)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen2.5-32B-Instruct --port 8000 --tensor-parallel-size 1
```

### 4. GRPO Training

```bash
# verl uses sglang for actor (Qwen3-8B) rollout, calls port 8000 for fact/citation reward
bash verl/scripts/run_grpo.sh
```

### 5. Evaluation

```bash
bash eval/run_eval.sh
```

---

## Hardware Requirements

| Stage | GPU | Memory |
|-------|-----|--------|
| SFT | 4× A100 80GB  |
| GRPO | 4× A100 80GB | 

---

## Dependencies

- **Models**: Qwen3-8B (actor), Qwen2.5-32B-Instruct (reward judge)
- **Frameworks**: verl, sglang, LlamaFactory, vllm
- **APIs**: SiliconFlow (claim verification), Tavily/Bocha (web search)
- **Data**: KLCF dataset (14,358 → 2,918 SFT → 1,776 GRPO)
