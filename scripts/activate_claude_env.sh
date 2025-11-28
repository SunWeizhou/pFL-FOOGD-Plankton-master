#!/usr/bin/env bash
# 在当前 shell 会话中临时加载项目的 .claude_env，然后执行传入的命令（或打开一个交互 shell）
# 用法：
#   bash scripts/activate_claude_env.sh            -> 进入一个新的 shell（环境已加载）
#   bash scripts/activate_claude_env.sh your_cmd   -> 在加载了 env 的环境中执行 your_cmd

set -euo pipefail

PROJECT_ROOT="/home/dell7960/桌面/FedRoD/pFL-FOOGD-Plankton-master-main"
ENV_FILE="$PROJECT_ROOT/.claude_env"

if [ ! -f "$ENV_FILE" ]; then
  echo "找不到 $ENV_FILE。请确认在项目根目录下存在 .claude_env 文件。"
  exit 1
fi

# source env 文件到当前进程
# 注意：不要在脚本中回显或打印敏感变量
. "$ENV_FILE"

if [ $# -eq 0 ]; then
  # 交互 shell
  echo "环境已加载：你现在可以直接运行 claude 或其它依赖这些变量的工具。输入 exit 退出。"
  exec bash --noprofile --norc
else
  # 执行传入的命令（作为子进程）
  exec "$@"
fi
