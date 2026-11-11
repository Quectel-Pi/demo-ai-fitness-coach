#!/usr/bin/env python3
"""板端诊断: 主干 QNN EP(HTP) + 检测头 CPU 的精度与耗时对比。

用法: python3 tools/diag_qnn.py <full.onnx> <backbone_qdq.onnx> <head.onnx> [图片目录]
"""
import glob
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

QNN_LIB = "/usr/lib/python3/dist-packages/onnxruntime_qnn"
SIZE = 640


def pre(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = SIZE / max(h, w)
    nw, nh = round(w * s), round(h * s)
    c = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    c.paste(img.resize((nw, nh), Image.BILINEAR),
            ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return np.asarray(c, dtype=np.float32).transpose(2, 0, 1)[None]


def main():
    full, bq, hd = sys.argv[1], sys.argv[2], sys.argv[3]
    ids = sys.argv[4] if len(sys.argv) > 4 else "/home/q/h1work/calib"
    files = sorted(glob.glob(ids + "/*.jpg"))[:8]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    cpu = ort.InferenceSession(full, so, providers=["CPUExecutionProvider"])
    head = ort.InferenceSession(hd, so, providers=["CPUExecutionProvider"])

    ort.register_execution_provider_library(
        "QNNExecutionProvider", QNN_LIB + "/libonnxruntime_providers_qnn.so")
    devs = [d for d in ort.get_ep_devices()
            if d.ep_name == "QNNExecutionProvider"]
    so_qnn = ort.SessionOptions()
    so_qnn.add_provider_for_devices(
        devs, {"backend_path": QNN_LIB + "/libQnnHtp.so",
               "htp_performance_mode": "burst"})
    t0 = time.time()
    bb = ort.InferenceSession(bq, sess_options=so_qnn)
    print(f"[qnn] init {time.time() - t0:.1f}s providers={bb.get_providers()}",
          flush=True)
    bb_in = bb.get_inputs()[0].name
    hd_in = [i.name for i in head.get_inputs()]

    def bench(sess, feed, n=1):
        sess.run(None, feed)
        t = time.time()
        for _ in range(n):
            out = sess.run(None, feed)
        return out, (time.time() - t) / n * 1000

    for f in files:
        x = pre(f)
        o1, t1 = bench(cpu, {"images": x}, 2)
        feats, tb = bench(bb, {bb_in: x}, 10)
        o2, th = bench(head, dict(zip(hd_in, feats)), 10)
        a = o1[0][0]
        b = o2[0][0]
        i = int(a[4].argmax())
        k1 = a[5:, i].reshape(17, 3)
        k2 = b[5:, i].reshape(17, 3)
        d = np.linalg.norm(k1[:, :2] - k2[:, :2], axis=1)
        print(f"{f.split('/')[-1]:10s} full {t1:6.1f}ms | bb {tb:6.1f}ms "
              f"head {th:5.1f}ms | conf {a[4,i]:.3f}->{b[4,i]:.3f} "
              f"kp mean {d.mean():5.2f} max {d.max():5.2f}px "
              f"kpconf {np.abs(k1[:,2]-k2[:,2]).max():.3f}", flush=True)


if __name__ == "__main__":
    main()
