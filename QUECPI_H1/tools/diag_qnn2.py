#!/usr/bin/env python3
"""板端诊断: 整模型 QNN EP(HTP) 精度/耗时（4 输出模型）。"""
import glob, sys, time
import numpy as np, onnxruntime as ort
from PIL import Image
QNN_LIB = "/usr/lib/python3/dist-packages/onnxruntime_qnn"
SIZE = 640
def pre(path):
    img = Image.open(path).convert("RGB"); w, h = img.size
    s = SIZE / max(h, w); nw, nh = round(w * s), round(h * s)
    c = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    c.paste(img.resize((nw, nh), Image.BILINEAR), ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return (np.asarray(c, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
def main():
    qmodel = sys.argv[1]; ids = sys.argv[2] if len(sys.argv) > 2 else "/home/q/h1work/calib"
    files = sorted(glob.glob(ids + "/*.jpg"))[:6]
    ort.register_execution_provider_library("QNNExecutionProvider", QNN_LIB + "/libonnxruntime_providers_qnn.so")
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    so = ort.SessionOptions()
    so.add_provider_for_devices(devs, {"backend_path": QNN_LIB + "/libQnnHtp.so", "htp_performance_mode": "burst"})
    t0 = time.time()
    s = ort.InferenceSession(qmodel, sess_options=so)
    print(f"[qnn] init {time.time()-t0:.1f}s providers={s.get_providers()}", flush=True)
    print("outputs", [(o.name, o.shape) for o in s.get_outputs()], flush=True)
    x = pre(files[0])
    s.run(None, {"images": x})
    t = time.time()
    for _ in range(10): out = s.run(None, {"images": x})
    print(f"latency {(time.time()-t)/10*1000:.1f}ms", flush=True)
    o = out
    i = int(o[0][0, 0].argmax())
    print("conf", float(o[0][0, 0, i]), "anchor", i, flush=True)
    print("kp", np.round(np.stack([o[1][0, :, 0, i], o[1][0, :, 1, i], o[2][0, :, i]], 1), 1)[:6], flush=True)
if __name__ == "__main__":
    main()
