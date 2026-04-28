# DeepCiteFact

基于 GRPO 强化学习的长文本事实性增强训练框架，通过多维奖励函数同时优化事实准确性、引用规范性和格式完整性。

**核心结果：**
- Citation Precision: 0.189 (SFT) → 0.712 (GRPO), **+276%**
- Avg Correct Citations: 0.765 → 2.9, **+279%**
- Fact Score: 0.719 → 0.785

---

## 项目结构

```
DeepCiteFact/
├── README.md
├── requirements.txt
├── autodl_deploy_guide.txt    # AutoDL 部署指南
├── run_api_version.sh         # API 服务启动脚本
├── LlamaFactory/               # SFT 训练 (Qwen3-8B)
│   └── train/sft.yaml
├── data/                       # 训练数据集
│   ├── sft_trace_filter.jsonl           # 过滤后 SFT 轨迹 (2,918 条)
│   └── rl_train_data_filter_grpo.jsonl  # GRPO 硬样本 (1,776 条)
├── eval/                       # 评估脚本 (citation + fact metrics)
│   ├── citation_eval.py
│   ├── fact_eval.py
│   └── get_response.py
├── verl/                       # verl 框架 + 自定义 reward manager
│   ├── verl/workers/reward_manager/
│   │   ├── custom.py           # CustomRewardManager (4 维奖励函数)
│   │   └── utils/             # reward 计算工具
│   ├── verl/examples/sglang_multiturn/  # GRPO 训练配置
│   │   └── config/
│   │       ├── search_grpo.yaml
│   │       └── tool_config/custom_tool_config.yaml
│   ├── scripts/run_grpo.sh    # GRPO 训练启动脚本
│   ├── tensorboard_log/        # 训练过程可视化日志
│  
└── to_hf/                      # checkpoint 导出工具
    ├── model_merge.sh
    └── legacy_model_merger.py
```

---

## 四维奖励函数

```
R_total = 0.5 × R_fact + 0.35 × R_cite + 0.1 × R_search + 0.05 × R_format
```

| 组件 | 方法 | 权重 |
|------|------|------|
| `R_fact` | Qwen2.5-32B 原子声明验证 | 0.50 |
| `R_cite` | URL 真实性 (0.4) + 语义 F1 (0.6) | 0.35 |
| `R_search` | 有效搜索次数 / 5 | 0.10 |
| `R_format` | 标签闭合验证 | 0.05 |

---

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
cd verl && pip install -e . && cd ..
pip install "sglang[all]==0.4.9.post6"
```

### 2. SFT 训练

```bash
cd LlamaFactory
llamafactory-cli train train/sft.yaml
```

### 3. GRPO 训练

```bash
# 启动 reward 计算服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen2.5-32B-Instruct --port 8000 --tensor-parallel-size 1

# 运行 GRPO
bash verl/scripts/run_grpo.sh
```

### 4. 评估

```bash
bash eval/run_eval.sh
```

---

## 硬件要求

| 阶段 | GPU | 
|------|-----|
| SFT | A100 80GB|
| GRPO | 4× A100 80GB | 



---

## 依赖

- **模型**: Qwen3-8B (actor), Qwen2.5-32B-Instruct (reward judge)
- **框架**: verl, sglang, LlamaFactory, vllm
- **API**: SiliconFlow (声明验证), Tavily/Bocha (搜索)
- **数据**: KLCF 数据集 (14,358 → 2,918 SFT → 1,776 GRPO)
