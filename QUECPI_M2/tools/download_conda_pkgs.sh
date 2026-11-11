#!/bin/bash
# 从清华镜像批量下载 conda 依赖闭包（由 resolve_conda_deps.py 生成的清单）
# 所有文件只落到本目录，不影响宿主机环境
set -u
BASE_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-aarch64"
DEST="$(cd "$(dirname "$0")/.." && pwd)/packages/conda"
LIST="${1:-pkgs.txt}"   # 依赖清单（python3 tools/resolve_conda_deps.py > pkgs.txt 生成）
mkdir -p "$DEST"

cd "$DEST" || exit 1
fail=0
while IFS=$'\t' read -r fn _size; do
    [ -f "$fn" ] && continue
    if ! curl -sf --retry 3 --max-time 300 "${BASE_URL}/${fn}" -o "$fn"; then
        echo "FAILED: $fn" >&2
        fail=1
    fi
done < "$LIST"
exit $fail
