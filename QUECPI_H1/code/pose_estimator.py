#!/usr/bin/env python3
"""YOLO11n-pose 推理封装（17 个 COCO 关键点），H1(QCS6490) 双后端:

  - qnn  : 主干走 QNN EP / HTP v68 NPU（u8 量化，约 9ms）
           + 检测头（DFL 解码）在 CPU float（约 36ms，保精度）
  - onnx : 整模型 CPU onnxruntime（约 100ms，兜底/离线验证）

输入: BGR numpy 帧 (cv2)
输出: (17,3) 关键点 [x, y, conf]（原图像素坐标）；无人体时 None。
注意: 本模型 ONNX 输入要求 0~1 归一化（不含内建 /255）。
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

# 板上 QAIRT 运行时自带的 QNN EP（插件式）+ HTP 后端库
QNN_PKG = os.environ.get("QNN_PKG", "/usr/lib/python3/dist-packages/onnxruntime_qnn")
QNN_EP_LIB = os.environ.get("QNN_EP_LIB",
                            str(Path(QNN_PKG) / "libonnxruntime_providers_qnn.so"))
QNN_HTP_LIB = os.environ.get("QNN_HTP_LIB",
                             str(Path(QNN_PKG) / "libQnnHtp.so"))
_EP_REGISTERED = False


def _register_qnn_ep():
    """注册 QNN 插件 EP（进程内只需一次）"""
    global _EP_REGISTERED
    import onnxruntime as ort
    if _EP_REGISTERED or "QNNExecutionProvider" in ort.get_available_providers():
        _EP_REGISTERED = True
        return
    ort.register_execution_provider_library("QNNExecutionProvider", QNN_EP_LIB)
    _EP_REGISTERED = True


class PoseEstimator:
    def __init__(self, model_dir: str, backend: str = None, input_size: int = 640,
                 threads: int = 4):
        self.input_size = input_size
        self.backend = backend or os.environ.get("POSE_BACKEND", "qnn")
        model_dir = Path(model_dir)

        if self.backend == "qnn":
            self._init_qnn(model_dir, threads)
        else:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            self.session = ort.InferenceSession(
                str(model_dir / "yolo11n-pose.onnx"), so,
                providers=["CPUExecutionProvider"])
            self._input_name = self.session.get_inputs()[0].name

    # ---------------- QNN: 主干 NPU + 检测头 CPU ----------------
    def _init_qnn(self, model_dir: Path, threads: int):
        import onnxruntime as ort
        _register_qnn_ep()
        devs = [d for d in ort.get_ep_devices()
                if d.ep_name == "QNNExecutionProvider"]
        if not devs:
            raise RuntimeError("QNNExecutionProvider 未注册成功")
        so = ort.SessionOptions()
        # 关闭 QNN 会话线程池自旋: 自旋线程会抢占大核，使随后的 head CPU
        # 推理从 ~37ms 恶化到 ~67ms（实测 76ms/帧 -> 46ms/帧）
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_provider_for_devices(devs, {
            "backend_path": QNN_HTP_LIB,
            "htp_performance_mode": "burst",
        })
        self.bb = ort.InferenceSession(
            str(model_dir / "yolo11n-pose-backbone-qdq.onnx"), so)
        so_cpu = ort.SessionOptions()
        so_cpu.intra_op_num_threads = threads
        self.head = ort.InferenceSession(
            str(model_dir / "yolo11n-pose-head.onnx"), so_cpu,
            providers=["CPUExecutionProvider"])
        self._bb_in = self.bb.get_inputs()[0].name
        self._head_in = [i.name for i in self.head.get_inputs()]
        print(f"[pose] backend=qnn htp, bb_in={self._bb_in}, "
              f"head_in={self._head_in}", flush=True)

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
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        if self.backend == "qnn":
            feats = self.bb.run(None, {self._bb_in: blob})
            out = self.head.run(None, dict(zip(self._head_in, feats)))[0]
        else:
            out = self.session.run(None, {self._input_name: blob})[0]

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
