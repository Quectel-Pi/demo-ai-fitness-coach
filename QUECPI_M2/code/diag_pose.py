#!/usr/bin/env python3
"""板上诊断: 打印关键点与膝角，验证角度计算与关键点质量。"""
import sys
import time

sys.path.insert(0, "/root/ai-fitness-coach/code")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pose_estimator import PoseEstimator  # noqa: E402


def ang(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


est = PoseEstimator("/root/ai-fitness-coach/models", backend="rknn")
cap = cv2.VideoCapture(73, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(0.3)

names = ["nose", "lsho", "rsho", "lhip", "rhip", "lknee", "rknee", "lank", "rank"]
ids = [0, 5, 6, 11, 12, 13, 14, 15, 16]
t0 = time.time()
n = det = 0
while time.time() - t0 < 20:
    ok, f = cap.read()
    if not ok:
        continue
    n += 1
    f = cv2.flip(f, 1)
    k = est.estimate(f)
    if k is None:
        continue
    det += 1
    if det % 8 == 1:
        lh, rh = k[11][:2], k[12][:2]
        lk, rk = k[13][:2], k[14][:2]
        la, ra = k[15][:2], k[16][:2]
        pts = " ".join(f"{nm}=({k[j][0]:.0f},{k[j][1]:.0f},{k[j][2]:.2f})"
                       for nm, j in zip(names, ids))
        print(f"L膝={ang(lh, lk, la):.0f} R膝={ang(rh, rk, ra):.0f} | {pts}",
              flush=True)
    time.sleep(0.1)

print(f"frames={n} detected={det} ({det/max(n,1)*100:.0f}%)", flush=True)
cap.release()
est.close()
