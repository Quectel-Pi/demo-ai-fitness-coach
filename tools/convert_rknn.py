#!/usr/bin/env python3
"""主机端：把 yolo11n-pose.onnx 转成 RK3576 NPU 的 .rknn（fp16，不量化保精度）。

依赖（隔离安装到 .build/rknn-env，不污染宿主机）:
  pip install --target .build/rknn-env <cp310 wheel>
  onnx==1.15.0 protobuf==4.25.3（rknn-toolkit2 2.3.2 兼容版）

用法: python3 tools/convert_rknn.py
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
WHEEL = BASE / ".build/rknn-host/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
ENV = BASE / ".build/rknn-env"
ONNX = BASE / "models/yolo11n-pose-op19.onnx"   # opset 19（rknn-toolkit2 上限）
OUT = BASE / "models/yolo11n-pose.rknn"

sys.path.insert(0, str(ENV))


def main():
    from rknn.api import RKNN as RKNN_API

    rknn = RKNN_API(verbose=False)
    # Ultralytics 归一化: x/255 -> mean=0, std=255；NHWC 由 runtime 处理
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3576",
    )
    if rknn.load_onnx(model=str(ONNX)) != 0:
        raise SystemExit("load_onnx failed")
    # fp16 不量化：关键点坐标对量化敏感，保精度
    if rknn.build(do_quantization=False) != 0:
        raise SystemExit("build failed")
    if rknn.export_rknn(str(OUT)) != 0:
        raise SystemExit("export failed")
    rknn.release()
    print(f"[ok] {OUT} ({OUT.stat().st_size // 1048576} MB)")


if __name__ == "__main__":
    main()
