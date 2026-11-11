#!/bin/bash
# 主机端：把 packages/ 下的 conda 包与 wheel 解压合并成可部署的 runtime/
# 产物: runtime/py (PYTHONPATH) + runtime/lib (LD_LIBRARY_PATH) + runtime/share
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
PKGS="$BASE/packages"
CONDA_DIR="$PKGS/conda"
RUNTIME="$BASE/runtime"
# 解包工作目录：默认项目内 .tmp-unpack（构建完自动清理；可用 TMPDIR_WORK 覆盖，
# 注意 /home/q/work 下沙箱会拦截 rm/cp，若覆盖请用可写可删的目录）
WORK="${TMPDIR_WORK:-$BASE/.tmp-unpack}"

rm -rf "$RUNTIME" "$WORK"
mkdir -p "$WORK" "$RUNTIME/py" "$RUNTIME/lib" "$RUNTIME/share"

unpack_conda() {
    local f="$1" d="$2"
    mkdir -p "$d"
    # .conda 是 zip 容器，内含 pkg-*.tar.zst（主体）
    python3 -m zipfile -e "$f" "$d"
    local inner
    inner="$(ls "$d"/pkg-*.tar.zst 2>/dev/null | head -1 || true)"
    if [ -n "$inner" ]; then
        zstd -dc "$inner" | tar -xf - -C "$d"
    fi
}

# 1) conda 包
for f in "$CONDA_DIR"/*.conda "$CONDA_DIR"/*.tar.bz2; do
    [ -f "$f" ] || continue
    d="$WORK/$(basename "$f")"
    case "$f" in
        *.conda) unpack_conda "$f" "$d" ;;
        *.tar.bz2) mkdir -p "$d" && tar -xjf "$f" -C "$d" ;;
    esac
    # site-packages -> runtime/py
    if [ -d "$d/lib/python3.13/site-packages" ]; then
        cp -a "$d/lib/python3.13/site-packages/." "$RUNTIME/py/"
    fi
    # lib/* (除 python*) -> runtime/lib
    if [ -d "$d/lib" ]; then
        (cd "$d/lib" && find . -maxdepth 1 -mindepth 1 ! -name 'python*' \
            -exec cp -a {} "$RUNTIME/lib/" \;)
    fi
    # share -> runtime/share (Qt translations 等)
    if [ -d "$d/share" ]; then
        cp -a "$d/share/." "$RUNTIME/share/"
    fi
    echo "[ok] $(basename "$f")"
done

# 2) wheel（onnxruntime / protobuf / flatbuffers / packaging；numpy 用板上系统自带）
for w in "$PKGS"/*.whl; do
    [ -f "$w" ] || continue
    case "$(basename "$w")" in numpy-*) echo "[skip] $(basename "$w") (板上已有)"; continue ;; esac
    python3 -m zipfile -e "$w" "$WORK/wheel-$(basename "$w" .whl)"
    d="$WORK/wheel-$(basename "$w" .whl)"
    cp -a "$d/." "$RUNTIME/py/"
    echo "[ok] $(basename "$w")"
done

echo "---- runtime 构建完成 ----"
du -sh "$RUNTIME"
rm -rf "$WORK"
