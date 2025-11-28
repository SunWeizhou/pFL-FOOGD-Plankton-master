#!/usr/bin/env bash
# 安全地将项目内 .claude_env 添加到用户 ~/.bashrc 中（如果尚未添加）
# 使用方法: bash scripts/add_claude_env_to_bashrc.sh

set -euo pipefail

PROJECT_ROOT="/home/dell7960/桌面/FedRoD/pFL-FOOGD-Plankton-master-main"
ENV_FILE="$PROJECT_ROOT/.claude_env"
BASHRC="$HOME/.bashrc"
MARKER_START="# >>> claude env (added by project) >>>"
MARKER_END="# <<< claude env (added by project) <<<"

if [ ! -f "$ENV_FILE" ]; then
  echo "找不到 $ENV_FILE，请确认脚本在正确的工作区运行。"
  exit 1
fi

# 检查是否已存在相同的标记
if grep -Fq "$MARKER_START" "$BASHRC" 2>/dev/null || grep -Fq "$ENV_FILE" "$BASHRC" 2>/dev/null; then
  echo "你的 $BASHRC 已经包含对该 env 的引用，跳过追加。"
  exit 0
fi

# 备份 bashrc
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="$BASHRC.backup.$TIMESTAMP"
cp "$BASHRC" "$BACKUP" && echo "已备份 $BASHRC 到 $BACKUP"

# 追加标记和 source 行
cat >> "$BASHRC" <<EOF

$MARKER_START
# 项目 claude API 环境变量（仅在此项目需要时启用）
if [ -f "$ENV_FILE" ]; then
  # 不使用 export PATH 之类的敏感操作，直接 source 环境文件
  . "$ENV_FILE"
fi
$MARKER_END
EOF

echo "已把 source 添加到 $BASHRC。要生效请运行： source $BASHRC 或重新打开终端/重启 VS Code。"
exit 0
