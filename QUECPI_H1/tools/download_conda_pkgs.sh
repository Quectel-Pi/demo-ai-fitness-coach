#!/bin/bash
# 从清华镜像批量下载 conda 依赖闭包（由 resolve_conda_deps.py 生成的清单）
# 所有文件只落到本目录，不影响宿主机环境
# 用法: tools/download_conda_pkgs.sh [清单文件]
#   清单文件缺省优先级: $FITNESS_PKGS_LIST → 项目根目录 pkgs.txt
#   （与 README 部署步骤 "python3 tools/resolve_conda_deps.py > pkgs.txt" 配套）
set -u
BASE_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-aarch64"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/packages/conda"
LIST="${1:-${FITNESS_PKGS_LIST:-$ROOT/pkgs.txt}}"
if [ ! -s "$LIST" ]; then
    echo "!! 清单不存在或为空: $LIST" >&2
    echo "   先执行: python3 tools/resolve_conda_deps.py > pkgs.txt" >&2
    exit 1
fi
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

# 清理不在清单里的旧包：清单更新后（依赖解析修复导致版本变化），
# 旧包必须清掉，否则 make_runtime 会把整个目录全部解包，过期 so 混进 runtime
removed=0
for f in ./*.conda ./*.tar.bz2; do
    [ -e "$f" ] || continue
    if ! cut -f1 "$LIST" | grep -qxF "$(basename "$f")"; then
        echo "[prune] $(basename "$f")"
        rm -f "$f"
        removed=$((removed + 1))
    fi
done
echo "[prune] removed $removed stale package(s)"

exit $fail
