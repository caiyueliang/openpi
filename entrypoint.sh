#!/bin/bash
set -e

# 设置默认值
MODEL_PATH=${MODEL_PATH:-"/home/caiyueliang/models/ckpt/pi05_libero"}
POLICY_CONFIG=${POLICY_CONFIG:-"pi05_libero"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8080}

# 打印配置信息（可选）
echo "============================================================="
echo "[MODEL_PATH] $MODEL_PATH"
echo "[POLICY_CONFIG] $POLICY_CONFIG"
echo "[HOST] $HOST"
echo "[PORT] $PORT"
echo "============================================================="

# 启动 Python 服务
uv run scripts/serve_policy.py policy:checkpoint --policy.config=${POLICY_CONFIG} --policy.dir=${MODEL_PATH}