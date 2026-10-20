#!/usr/bin/env python3
"""判定素材是深蹲还是弓步蹲: L/R 膝角对称性分析（板上用应用自带模型跑）。
深蹲: 双膝同步弯曲, L/R 膝角差 < 20°
弓步/分腿蹲: 前膝深后膝浅, L/R 膝角差 > 30°
用法: python3 check_action.py <video> [stride]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np

from pose_estimator import PoseEstimator

def angle(a, k, b):
    """膝角: 髋-膝-踝三点角(180°=伸直)"""
    v1, v2 = a - k, b - k
    cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

video = sys.argv[1]
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 6   # 每 6 帧采样
est = PoseEstimator(str(Path(__file__).resolve().parent.parent / "models"))

cap = cv2.VideoCapture(video)
rows, i = [], 0
while True:
    ok = cap.grab()
    if not ok:
        break
    if i % stride == 0:
        ok, frame = cap.retrieve()
        if not ok:
            break
        if frame.shape[0] > 480:
            s = 480.0 / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * s), 480))
        kpts = est.estimate(frame)
        if kpts is not None:
            # COCO: 11 L髋 12 R髋 13 L膝 14 R膝 15 L踝 16 R踝
            lk = angle(kpts[11, :2], kpts[13, :2], kpts[15, :2])
            rk = angle(kpts[12, :2], kpts[14, :2], kpts[16, :2])
            ank = np.linalg.norm(kpts[15, :2] - kpts[16, :2])  # 双踝距离
            hip_y = (kpts[11, 1] + kpts[12, 1]) / 2
            rows.append((i, lk, rk, abs(lk - rk), ank, hip_y))
    i += 1
cap.release()
est.close()

if not rows:
    print("NO POSE DETECTED")
    sys.exit(1)

diffs = [r[3] for r in rows]
mins = [min(r[1], r[2]) for r in rows]
print(f"frames sampled={len(rows)}")
print(f"L/R knee angle diff: mean={np.mean(diffs):.0f}° max={np.max(diffs):.0f}° "
      f"p90={np.percentile(diffs, 90):.0f}°")
print(f"min-knee (deepest bend): min={np.min(mins):.0f}° mean={np.mean(mins):.0f}°")
# 判定: 深蹲在整个动作周期双膝对称; 弓步蹲在最低点不对称
bottom = sorted(rows, key=lambda r: min(r[1], r[2]))[: max(3, len(rows) // 5)]
bdiff = np.mean([r[3] for r in bottom])
print(f"at bottom phase: L/R diff mean={bdiff:.0f}°")
verdict = "SQUAT(深蹲)" if bdiff < 20 and np.percentile(diffs, 90) < 35 else \
          ("LUNGE/SPLIT(弓步/分腿蹲)" if bdiff > 30 else "AMBIGUOUS(临界)")
print(f"VERDICT: {verdict}")
# 逐帧明细(最低点附近)
print("\nbottom frames (frame, L膝, R膝, diff, 踝距, 髋y):")
for r in sorted(rows, key=lambda r: min(r[1], r[2]))[:8]:
    print(f"  f{r[0]:4d}  L={r[1]:5.0f} R={r[2]:5.0f} diff={r[3]:4.0f} ank={r[4]:4.0f} hipy={r[5]:4.0f}")
