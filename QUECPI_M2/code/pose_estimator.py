#!/usr/bin/env python3
"""YOLO11n-pose 推理封装（17 个 COCO 关键点），双后端:
  - rknn : RK3576 NPU（fp16，~25ms/帧，优先）
  - onnx : CPU onnxruntime（~300ms/帧，兜底）

输入: BGR numpy 帧 (cv2)
输出: (17,3) 关键点 [x, y, conf]（原图像素坐标）；无人体时 None。
"""
import os
from pathlib import Path

import numpy as np
import cv2

# COCO 17 关键点索引
KP = {
    "nose": 0,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# 骨架连线（UI 绘制）
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 5), (0, 6),
]


class PoseEstimator:
    def __init__(self, model_dir: str, backend: str = None, input_size: int = 640, threads: int = 4):
        self.input_size = input_size
        self.backend = backend or os.environ.get("POSE_BACKEND", "rknn")

        if self.backend == "rknn":
            from rknn_backend import RKNNRuntime
            model = str(Path(model_dir) / "yolo11n-pose.rknn")
            self.session = RKNNRuntime(model)
        else:
            import onnxruntime as ort
            model = str(Path(model_dir) / "yolo11n-pose.onnx")
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            self.session = ort.InferenceSession(
                model, so, providers=["CPUExecutionProvider"])
            self._input_name = self.session.get_inputs()[0].name
        self._is_ort = self.backend != "rknn"

    def _letterbox(self, img: np.ndarray):
        """等比缩放 + 居中填充到 input_size 方形。返回 (canvas_rgb, scale, pad_x, pad_y)"""
        h, w = img.shape[:2]
        s = self.input_size / max(h, w)
        nw, nh = round(w * s), round(h * s)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas[:, :, ::-1], s, pad_x, pad_y  # BGR -> RGB

    def estimate(self, img: np.ndarray, conf_thres: float = 0.35):
        canvas, s, px, py = self._letterbox(img)

        if self._is_ort:
            blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
            out = self.session.run(None, {self._input_name: blob})[0]
        else:
            nhwc = canvas[None].astype(np.float32)   # NHWC float32, NPU 自动量化
            out = self.session.run(nhwc)
            out = out.reshape(1, 56, -1) if out.ndim == 3 else out

        pred = out[0]                                # (56, 8400)
        conf = pred[4]
        best = int(conf.argmax())
        best_conf = float(conf[best])
        if best_conf < conf_thres:
            return None
        kpts = pred[5:, best].reshape(17, 3).copy()
        kpts[:, 0] = (kpts[:, 0] - px) / s
        kpts[:, 1] = (kpts[:, 1] - py) / s
        return kpts

    def close(self):
        if not self._is_ort and hasattr(self.session, "close"):
            self.session.close()
