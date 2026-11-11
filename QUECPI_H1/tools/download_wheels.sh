#!/bin/bash
# 下载 runtime/ 所需的 aarch64 wheel 包（make_runtime.sh 第 2 段会解包它们）
#   opencv-python-headless（cv2 采集与图像处理）
#   protobuf / flatbuffers（onnxruntime 依赖）
#   packaging（onnxruntime 依赖）
#   numpy 不下——板上系统 Python 自带，make_runtime 会主动跳过
# 用法: tools/download_wheels.sh        （下载到 packages/，可重复执行自动跳过已有）
# 说明: 用 pip download 拿 aarch64 manylinux wheel，不安装、不污染宿主机；
#       显式走清华镜像并加超时（默认源在部分网络下会长时间无响应）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/packages"
mkdir -p "$DEST"

python3 -m pip download \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --timeout 60 \
    --no-deps --only-binary=:all: \
    --platform manylinux2014_aarch64 --python-version 3.13 \
    -d "$DEST" \
    "opencv-python-headless==4.12.0.88" \
    "protobuf==7.36.2" \
    "flatbuffers==25.12.19" \
    "packaging==26.3"

echo "---- wheels 就绪 ----"
ls "$DEST"/*.whl
