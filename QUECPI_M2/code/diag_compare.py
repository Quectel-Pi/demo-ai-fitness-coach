#!/usr/bin/env python3
"""板上对比诊断: 同一帧 RKNN-NPU vs ONNX-CPU 的关键点输出对比。"""
import sys
import time

sys.path.insert(0, "/root/ai-fitness-coach/code")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rknn_backend import RKNNRuntime  # noqa: E402
import onnxruntime as ort  # noqa: E402


def letterbox(img, size=640):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = round(w * s), round(h * s)
    r = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = r
    return canvas[:, :, ::-1], s, px, py  # RGB


def parse(pred, s, px, py):
    """pred: (56, 8400) -> (17,3) 原图坐标 + 框"""
    conf = pred[4]
    b = int(conf.argmax())
    kpts = pred[5:, b].reshape(17, 3).copy()
    kpts[:, 0] = (kpts[:, 0] - px) / s
    kpts[:, 1] = (kpts[:, 1] - py) / s
    box = pred[:4, b]
    return kpts, float(conf[b]), box


rknn = RKNNRuntime("/root/ai-fitness-coach/models/yolo11n-pose.rknn")
so = ort.SessionOptions()
so.intra_op_num_threads = 4
ort_sess = ort.InferenceSession(
    "/root/ai-fitness-coach/models/yolo11n-pose.onnx", so,
    providers=["CPUExecutionProvider"])

cap = cv2.VideoCapture(73, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(0.3)

names = ["nose", "lsho", "rsho", "lhip", "rhip", "lknee", "rknee", "lank", "rank"]
ids = [0, 5, 6, 11, 12, 13, 14, 15, 16]

for i in range(30):
    ok, f = cap.read()
    if not ok:
        continue
    f = cv2.flip(f, 1)
    canvas, s, px, py = letterbox(f)

    # RKNN: NHWC float32 (内部 /255 由 std_values 处理)
    out_r = rknn.run(canvas[None].astype(np.float32)).reshape(56, -1)
    k_r, c_r, _ = parse(out_r, s, px, py)

    # ONNX CPU: NCHW float32 /255（原始模型输入 0-1）
    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out_o = ort_sess.run(None, {"images": blob})[0][0]
    k_o, c_o, _ = parse(out_o, s, px, py)

    if c_r > 0.5 or c_o > 0.5:
        print(f"--- frame {i}  boxconf rknn={c_r:.2f} cpu={c_o:.2f}")
        for nm, j in zip(names, ids):
            print(f"  {nm:6s} rknn=({k_r[j][0]:6.1f},{k_r[j][1]:6.1f},{k_r[j][2]:.2f})"
                  f"  cpu=({k_o[j][0]:6.1f},{k_o[j][1]:6.1f},{k_o[j][2]:.2f})"
                  f"  dxy=({k_r[j][0]-k_o[j][0]:+5.1f},{k_r[j][1]-k_o[j][1]:+5.1f})")
    time.sleep(0.3)

cap.release()
rknn.close()
