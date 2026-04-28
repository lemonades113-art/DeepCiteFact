#!/bin/bash
# DeepCiteFact API 版 - 一键启动脚本
# 用法：bash run_api_version.sh

set -x

cd /root/autodl-tmp/DeepCiteFact/verl

# 检查 API Key 是否设置
if [ "$SILICONFLOW_API_KEY" = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" ]; then
    echo "❌ 错误：请先修改 API Key！"
    echo "编辑 scripts/run_grpo.sh，找到第 12 行，替换为你的真实 API Key"
    exit 1
fi

# 检查模型是否存在
if [ ! -d "/root/autodl-tmp/models/Qwen3-8B" ]; then
    echo "❌ 错误：Qwen3-8B 模型不存在！"
    echo "请先下载模型："
    echo "  mkdir -p /root/autodl-tmp/models"
    echo "  cd /root/autodl-tmp/models"
    echo "  modelscope download Qwen/Qwen3-8B --local_dir ./Qwen3-8B"
    exit 1
fi

# 启动训练
echo "✅ 开始训练..."
bash scripts/run_grpo.sh
