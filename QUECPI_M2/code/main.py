#!/usr/bin/env python3
"""智能健身教练 主程序（PySide6，漫画风 UI）。

功能: USB 摄像头实时预览 + 17 关键点骨架绘制 + 深蹲/开合跳计次纠错 + 语音播报
运行: ./run.sh (板上) —— 需 runtime/ 环境与 models/ 模型
摄像头节点与窗口尺寸均为运行时自动发现（CAMERA_INDEX 环境变量可强制覆盖）。
"""
import argparse
import fcntl
import glob
import json
import math
import os
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (Qt, QThread, Signal, QTimer, QRectF, QPointF,
                            QPoint, QEvent)
from PySide6.QtGui import (QImage, QPixmap, QFont, QPainter, QColor, QPen,
                           QPolygonF, QLinearGradient)
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QButtonGroup, QFrame, QSizePolicy, QComboBox,
    QStyledItemDelegate, QStyleOptionViewItem, QStyle,
)

from pose_estimator import PoseEstimator, SKELETON
from exercise_analyzer import (EXERCISES, EXERCISE_NAMES,
                               set_lang as set_analyzer_lang)
from audio_player import AudioPlayer

MODEL_DIR = str(Path(__file__).resolve().parent.parent / "models")
# 动作参考演示视频目录（打包随 assets/ 分发）
REF_DIR = str(Path(__file__).resolve().parent.parent / "assets" / "refs")

GOOD_WORDS = ["good_1", "good_2", "good_3", "good_4"]
# 马步蹲计时里程碑语音（其余动作为数字播报）
HS_MILESTONES = {10: "hs_m10", 20: "hs_m20", 30: "hs_m30", 45: "hs_m45", 60: "hs_m60"}

# ---------------- demo 自动预演 ----------------
# --demo 模式: 按时间线自动操作 UI（模拟用户点击），配合 --video 素材与
# --record 帧录制，产出可后处理的录屏素材（帧序列 + meta.json 播报事件）。
RECORD_FPS = 10                          # 录屏帧率（grab 周期 100ms）

# 分镜时间线（深蹲实测模拟 + 功能导览）: 稳健蹲段循环输入（每 7.2s 循环 2 蹲，
# 双腿跟踪稳定、10fps 采样下 18/18 全计零拒绝），算法实时计次真实可见。
DEMO_SCRIPT = [
    (7.0,  "popup"),                      # 动作列表: 六种动作 + 每行▶演示按钮（核心镜头）
    (17.0, "play_ref", "squat"),          # 参考视频: 完整播标准深蹲示范（3 蹲，
                                          #  ~12s 后播完自动收起，不循环）
    (30.5, "popdown"),                    # 收起列表
    (31.0, "start"),                      # 开始训练: 即刻开闸，约 34s 起计次逐个增长
    (56.5, "pause"),                      # 暂停: 冻结约 7 次成绩
    (60.0, "resume"),                     # 继续: 恢复累计
    (67.0, "reset"),                      # 归零: 一键清零
    (69.0, "zoom_in"),                    # 变焦＋: 画面拉近
    (71.0, "zoom_reset"),                 # 变焦复位 1.0x
    (73.0, "lang"),                       # 中英切换: 界面+语音整体切英文
    (75.0, "lang"),                       # 切回中文
    (77.0, "finish"),                     # 结束训练（finish 语音后 2.5s 自动退出）
]

# ---------------- UI 文案双语 ----------------
UI = {
    "title":      {"zh": "★ 智能健身教练 ★", "en": "★ AI Fitness Coach ★"},
    "subtitle":   {"zh": "AI 姿态识别 · 实时计次 · 语音纠错",
                   "en": "AI Pose Tracking · Live Count · Voice Coach"},
    "lang_btn":   {"zh": "EN", "en": "中"},     # 按钮显示的是"将切换到"的语言
    "unit_reps":  {"zh": "次", "en": "reps"},
    "unit_secs":  {"zh": "秒", "en": "secs"},
    "ready":      {"zh": "准备就绪", "en": "Ready"},
    "good_job":   {"zh": "动作标准，继续保持 ★", "en": "Great form, keep going ★"},
    "zoom":       {"zh": "变焦 {v:.2f}x", "en": "Zoom {v:.2f}x"},
    "ref_btn":    {"zh": "▶ 演示", "en": "▶ Demo"},
    "ref_playing": {"zh": "参考演示 · {name}", "en": "Demo · {name}"},
    "ref_missing": {"zh": "该动作暂无演示视频", "en": "No demo video yet"},
    "cam_start":  {"zh": "启动摄像头中...", "en": "Starting camera..."},
    "finish_btn": {"zh": "结束训练", "en": "Finish"},
    "reset_btn":  {"zh": "归零", "en": "Reset"},
    "start_btn":  {"zh": "▶ 开始训练", "en": "▶ Start"},
    "pause_btn":  {"zh": "暂停", "en": "Pause"},
    "resume_btn": {"zh": "▶ 继续训练", "en": "▶ Resume"},
    "standby":    {"zh": "待机中 · 按「开始训练」后开始计次",
                   "en": "Standby · press Start to count"},
    "paused":     {"zh": "已暂停 · 按「继续训练」恢复",
                   "en": "Paused · press Resume to continue"},
    "problem":    {"zh": "！ {p}", "en": "! {p}"},
    # problems 关键词（分析器输出 key -> 展示文本）
    "knee_in":    {"zh": "膝内扣", "en": "knees caving in"},
    "trunk_lean": {"zh": "背部前倾", "en": "leaning forward"},
    "jj_arms_p":  {"zh": "手臂未过顶", "en": "arms not overhead"},
    "jj_feet_p":  {"zh": "双脚分不够开", "en": "feet not wide enough"},
    "no_person":  {"zh": "未检测到人", "en": "No person detected"},
}
LANG = os.environ.get("FITNESS_LANG", "zh")    # 初始语言可用环境变量指定


def ut(key: str, **kw) -> str:
    """UI 文案取当前语言"""
    return UI[key][LANG].format(**kw)


FONT_FAMILY = "sans-serif"  # main() 加载 CJK 字体后回填

# 板上 conda fontconfig 看不到系统字体 -> 直接注册字体文件到 Qt
CJK_FONTS = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def load_cjk_font():
    """注册板上中文字体，返回 family 名（供 QApplication 全局设置）"""
    from PySide6.QtGui import QFontDatabase
    for path in CJK_FONTS:
        if not os.path.exists(path):
            continue
        fid = QFontDatabase.addApplicationFont(path)
        if fid >= 0:
            fams = QFontDatabase.applicationFontFamilies(fid)
            if fams:
                print(f"[font] loaded {path} -> {fams[0]}", flush=True)
                return fams[0]
    print("[font] WARN: no CJK font loaded, text may show boxes", flush=True)
    return "sans-serif"


# ---------------- 摄像头自动发现 ----------------

def _v4l2_probe(path: str):
    """V4L2 能力查询。返回 (是否捕获设备, bus_info)。"""
    VIDIOC_QUERYCAP = 0x80685600
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            buf = bytearray(104)
            fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf)
        finally:
            os.close(fd)
    except OSError:
        return False, ""
    caps, dev_caps = struct.unpack_from("<II", buf, 84)
    eff = dev_caps if caps & 0x80000000 else caps   # V4L2_CAP_DEVICE_CAPS
    bus = bytes(buf[48:80]).split(b"\0", 1)[0].decode(errors="ignore")
    return bool(eff & 0x1), bus                     # V4L2_CAP_VIDEO_CAPTURE


def detect_camera_index() -> int:
    """扫描 /dev/video*，返回第一个能实际出图的捕获设备索引。

    USB 设备优先（避开 rkisp 等 platform 节点，后者 capture 位可能为真
    但 media pipeline 未配置时 read 会阻塞）。CAMERA_INDEX 可强制覆盖。"""
    if os.environ.get("CAMERA_INDEX"):
        idx = int(os.environ["CAMERA_INDEX"])
        print(f"[cam] CAMERA_INDEX={idx} (explicit)", flush=True)
        return idx

    import re
    devs = sorted((d for d in glob.glob("/dev/video*")
                   if re.fullmatch(r"/dev/video\d+", d)),
                  key=lambda p: int(p.rsplit("video", 1)[1]))
    candidates = []          # (usb优先级, 节点号, 路径)
    for d in devs:
        is_cap, bus = _v4l2_probe(d)
        if is_cap:
            n = int(d.rsplit("video", 1)[1])
            candidates.append((0 if bus.startswith("usb") else 1, n, d))
    candidates.sort()

    for _, idx, d in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            print(f"[cam] auto-detected {d}", flush=True)
            return idx
    print("[cam] WARN: no device yields frames, fallback to 0", flush=True)
    return 0


# ---------------- 漫画风计数控件 ----------------

class BurstLabel(QWidget):
    """爆炸星形 + 描边大数字；计数变化时脉冲放大一下。"""

    def __init__(self):
        super().__init__()
        self._value = 0
        self._pulse = 0.0
        self.setMinimumHeight(200)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def set_value(self, v: int, pulse: bool = True):
        v = int(v)
        if v == self._value:
            return
        self._value = v
        if pulse:
            self._pulse = 1.0
            self._timer.start()
        else:
            self.update()

    def _tick(self):
        self._pulse = max(0.0, self._pulse - 0.06)
        self.update()
        if self._pulse <= 0.0:
            self._timer.stop()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) * 0.44 * (1.0 + 0.12 * self._pulse)

        pts = []
        spikes = 12
        for i in range(spikes * 2):
            ang = i * math.pi / spikes - math.pi / 2
            rad = r if i % 2 == 0 else r * 0.68
            pts.append(QPointF(cx + rad * math.cos(ang),
                               cy + rad * math.sin(ang)))
        p.setPen(QPen(QColor("#2D3436"), 4))
        p.setBrush(QColor("#FFD93D"))
        p.drawPolygon(QPolygonF(pts))

        f = QFont(FONT_FAMILY)
        f.setPixelSize(int(min(w, h) * 0.38))
        f.setWeight(QFont.Black)
        p.setFont(f)
        text = str(self._value)
        rect = QRectF(0, 0, w, h)
        # 漫画描边字: 8 方向黑影 + 中心白字
        p.setPen(QColor("#2D3436"))
        for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3),
                       (-3, -3), (3, -3), (-3, 3), (3, 3)):
            p.drawText(rect.translated(dx, dy), Qt.AlignCenter, text)
        p.setPen(QColor("#FFFFFF"))
        p.drawText(rect, Qt.AlignCenter, text)


class Worker(QThread):
    """摄像头抓帧 + 推理 + 动作分析线程"""
    frame_ready = Signal(object)          # 带骨架绘制的 BGR 帧
    status_ready = Signal(object, object, list, float)  # count, stage, problems, fps

    def __init__(self, video: str = None):
        super().__init__()
        self.video = video                  # 视频文件路径（demo 录制用；None=摄像头）
        self.running = False
        self.exercise = "squat"
        self.zoom = 1.0                    # 数字变焦 0.5~3.0（中心裁剪）
        self._lock = threading.Lock()
        self.estimator = None
        self.analyzer = None
        self.audio = None  # 由主线程注入
        self.count = 0
        self.gated = True   # 训练门闸: True=待机/暂停（只预览，事件与计数全拦截）

    def set_zoom(self, z: float):
        with self._lock:
            self.zoom = max(0.5, min(3.0, round(z * 4) / 4))  # 步进 0.25

    def get_zoom(self) -> float:
        with self._lock:
            return self.zoom

    def set_exercise(self, name: str):
        with self._lock:
            self.exercise = name
            self.analyzer = EXERCISES[name]["cls"]()
            self.count = 0

    def reset_analyzer(self):
        with self._lock:
            self.analyzer = EXERCISES[self.exercise]["cls"]()
            self.count = 0

    def set_training(self, on: bool):
        """训练门闸: True=开始/继续 —— 换新分析器清瞬态（票数/基线），
        但保留已累计次数，暂停期间的干扰不会带进恢复后的记录。
        马步蹲无 count 属性（计时在 seconds），恢复 = 重新计时（休息语义）。"""
        with self._lock:
            self.gated = not on
            if on:
                fresh = EXERCISES[self.exercise]["cls"]()
                if hasattr(fresh, "count"):
                    fresh.count = self.count
                self.analyzer = fresh

    def run(self):
        self.estimator = PoseEstimator(MODEL_DIR)

        if self.video:
            # 视频文件模式（demo 录制）: 以墙上时钟为基准对齐视频时间轴，
            # 处理帧率低于视频帧率时用 grab() 跳帧，动作节奏与真实时间一致；
            # 素材播完自动循环回放
            cap = cv2.VideoCapture(self.video)
            if not cap.isOpened():
                print(f"[video] FAIL: cannot open {self.video}", flush=True)
                return
            vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[video] {self.video} @ {vfps:.1f}fps x {total} frames", flush=True)
            t_vid0 = time.monotonic()
            next_frame = 0                    # 下一次 grab() 将读取的帧号
        else:
            idx = detect_camera_index()
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)    # 通用后端兜底
            # 打印实际协商分辨率: 换 USB 摄像头时便于确认（驱动就近落到支持值）
            print(f"[cam] opened idx={idx} @ "
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}", flush=True)
        self.running = True
        t_prev = time.monotonic()
        fps_smooth = 0.0
        no_person_frames = 0

        while self.running and cap.isOpened():
            if self.video:
                # 时间对齐: 目标帧 = 经过时间 * 视频帧率（处理慢于视频帧率时跳帧）
                target = int((time.monotonic() - t_vid0) * vfps)
                if target >= total:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 素材播完 -> 循环回放
                    t_vid0 = time.monotonic()
                    target, next_frame = 0, 0
                ok = True
                while ok and next_frame <= target:
                    ok = cap.grab()                        # 只解码不取出，跳帧开销小
                    next_frame += 1
                if not ok:
                    time.sleep(0.02)
                    continue
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue
                if frame.shape[0] > 480:                  # 大分辨率素材等比缩到高 480，
                    s = 480.0 / frame.shape[0]             # 降低推理 letterbox 开销（几何不失真）
                    frame = cv2.resize(
                        frame, (int(frame.shape[1] * s), 480),
                        interpolation=cv2.INTER_AREA)
            else:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

            # 镜像预览（自拍模式）：用户正对摄像头时左右符合直觉。
            # 骨架在翻转后的帧上计算绘制，坐标自洽。
            frame = cv2.flip(frame, 1)

            # 数字变焦：z>1 中心裁剪放大（推理与骨架都在放大后的帧上）；
            # z<1 缩小加黑边，仅用于确认取景范围——摄像头 640x480 定焦定视野，
            # 缩小并不能看到镜头没拍到的内容（脚没入镜请后退或下调摄像头角度）
            z = self.get_zoom()
            h, w = frame.shape[:2]
            if z > 1.01:
                nw, nh = int(w / z), int(h / z)
                x0, y0 = (w - nw) // 2, (h - nh) // 2
                frame = cv2.resize(
                    frame[y0:y0 + nh, x0:x0 + nw], (w, h),
                    interpolation=cv2.INTER_LINEAR)
            elif z < 0.99:
                nw, nh = int(w * z), int(h * z)
                x0, y0 = (w - nw) // 2, (h - nh) // 2
                small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                frame[:] = (16, 20, 24)
                frame[y0:y0 + nh, x0:x0 + nw] = small

            kpts = self.estimator.estimate(frame)
            with self._lock:
                analyzer = self.analyzer
            fb = analyzer.update(kpts)

            if self.gated:
                # 待机/暂停: 只预览。事件与计数全部拦截，任何干扰不进记录
                fb.event = None
                fb.count = self.count
            else:
                # 每帧同步计数（马步蹲秒数无事件也持续增长）
                self.count = fb.count
                if fb.event == "count":
                    c = fb.count
                    if self.exercise == "horse_stance":
                        # 计时模式: 只在里程碑秒数播报
                        wav = HS_MILESTONES.get(c)
                        if wav:
                            self.audio.play(wav)
                    else:
                        if c % 10 == 0 and c <= 20 and c > 0:
                            self.audio.play(f"milestone_{c}")
                        elif 1 <= c <= 30:
                            self.audio.play(f"num_{c}")
                        if c in (5, 15, 25):
                            self.audio.play(GOOD_WORDS[c // 5 % 4])
                elif fb.event:
                    self.audio.play(fb.event, cooldown=True)

            # 无人: 静默等待，不催促；人从无到有的瞬间说一次"看到你了"
            # （待机/暂停期间不累计 —— 门闸关着不打扰）
            if kpts is None:
                if not self.gated:
                    no_person_frames += 1
            else:
                if no_person_frames >= 45:      # 之前至少 ~1.5s 没人
                    self.audio.play("person_back", cooldown=True)
                no_person_frames = 0

            # 绘制骨架
            if kpts is not None:
                self._draw(frame, kpts)

            now = time.monotonic()
            dt = now - t_prev
            t_prev = now
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if dt > 0 else fps_smooth

            self.frame_ready.emit(frame)
            self.status_ready.emit(fb.count, fb.stage, fb.problems, fps_smooth)

        cap.release()

    @staticmethod
    def _draw(frame: np.ndarray, k: np.ndarray):
        """漫画风骨架: 白色粗描边 + 青色主线 + 黄色关节黑环"""
        pts = {i: (int(x), int(y)) for i, (x, y, c) in enumerate(k) if c > 0.5}
        for i, j in SKELETON:
            if i in pts and j in pts:
                a, b = pts[i], pts[j]
                cv2.line(frame, a, b, (255, 255, 255), 7, cv2.LINE_AA)
                cv2.line(frame, a, b, (255, 170, 40), 3, cv2.LINE_AA)
        for p in pts.values():
            cv2.circle(frame, p, 7, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 4, (40, 230, 250), -1, cv2.LINE_AA)

    def stop(self):
        self.running = False
        self.wait(3000)


def build_qss() -> str:
    """漫画风全局样式（在 CJK 字体加载后调用）。
    注意: 窗口背景不在这里画 —— 无框圆角窗口由 MainWindow.paintEvent 自绘"""
    return f"""
QLabel, QPushButton {{ font-family: '{FONT_FAMILY}'; }}

#panel {{
    background: #FFF8E1;
    border: 4px solid #2D3436;
    border-radius: 18px;
}}
#title {{
    background: #FF6B6B; color: white;
    border: 3px solid #2D3436; border-radius: 14px;
    padding: 8px 6px;
    font-size: 21px; font-weight: 800;
}}
#subtitle {{ color: #6D6875; font-size: 12px; }}

QPushButton {{
    color: white;
    border: 3px solid #2D3436;
    border-radius: 14px;
    font-size: 15px; font-weight: 800;
    padding: 6px;
}}
QPushButton:hover {{ background: #74C0FC; }}
#btn_finish {{ background: #FF6B6B; }}
#btn_finish:hover {{ background: #FF8787; }}
#btn_reset {{ background: #51CF66; }}
#btn_reset:hover {{ background: #69DB7C; }}
#btn_start {{ background: #4D96FF; }}
#btn_start:hover {{ background: #74C0FC; }}
#btn_start[mode="pause"] {{ background: #FF922B; }}
#btn_start[mode="pause"]:hover {{ background: #FFA94D; }}
#btn_lang {{
    background: #B197FC; color: white;
    border: 3px solid #2D3436; border-radius: 12px;
    font-size: 14px; font-weight: 800;
}}
#btn_lang:hover {{ background: #D0BFFF; }}
#btn_lang:pressed {{ background: #9775FA; }}

QComboBox {{
    font-family: '{FONT_FAMILY}'; color: white;
    background: #4D96FF;
    border: 3px solid #2D3436; border-radius: 14px;
    font-size: 16px; font-weight: 800;
    padding: 8px 14px;
}}
QComboBox:hover {{ background: #74C0FC; }}
QComboBox::drop-down {{ border: none; width: 36px; }}
QComboBox QAbstractItemView {{
    font-family: '{FONT_FAMILY}';
    background: #FFF8E1;
    border: 3px solid #2D3436;
    padding: 0;
    outline: none;
    font-size: 15px;
}}
QComboBox QAbstractItemView::item {{
    color: #2D3436; background: #FFF8E1;
    padding: 8px 14px; min-height: 30px;
}}
QComboBox QAbstractItemView::item:hover {{ background: #FFE866; }}
QComboBox QAbstractItemView::item:selected {{ background: #FF6B6B; color: white; }}
#unit_label {{
    color: #2D3436; font-size: 16px; font-weight: 800;
}}

#zoom_btn {{
    background: #FFD93D; color: #2D3436;
    border-radius: 20px; font-size: 16px;
}}
#zoom_btn:hover {{ background: #FFE866; }}
#zoom_label {{ color: #2D3436; font-weight: 700; font-size: 13px; }}

#video {{
    background: #101418; color: #9AD1D4;
    border: 4px solid #2D3436; border-radius: 16px;
    font-size: 16px;
}}
#stage {{
    background: #4D96FF; color: white;
    border: 2px solid #2D3436; border-radius: 12px;
    padding: 5px; font-size: 14px; font-weight: 700;
}}
#fps {{ color: #8D99AE; font-size: 12px; }}
"""


PROBLEM_QSS = {
    "red": ("background: white; color: #E63946; border: 3px solid #2D3436;"
            " border-radius: 12px; padding: 6px; font-size: 15px;"
            " font-weight: 800;"),
    "green": ("background: white; color: #2B9348; border: 3px solid #2D3436;"
              " border-radius: 12px; padding: 6px; font-size: 15px;"
              " font-weight: 800;"),
}


class ComboItemDelegate(QStyledItemDelegate):
    """下拉条目渲染: 强制深色文字。

    Qt 已知行为——QComboBox 自身的 color(白) 会传播到弹层条目，
    QAbstractItemView / ::item 的 QSS 规则在部分版本无法覆盖，
    因此改由 delegate 直接接管底色与文字绘制（确定性生效）。"""

    BG = QColor("#FFF8E1")
    BG_HOVER = QColor("#FFE866")
    BG_SEL = QColor("#FF6B6B")
    FG = QColor("#2D3436")
    FG_SEL = QColor("white")

    def paint(self, painter: QPainter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        selected = bool(opt.state & QStyle.State_Selected)
        hovered = bool(opt.state & QStyle.State_MouseOver)
        painter.fillRect(opt.rect, self.BG_SEL if selected
                         else (self.BG_HOVER if hovered else self.BG))
        painter.setPen(self.FG_SEL if selected else self.FG)
        painter.setFont(opt.font)
        painter.drawText(opt.rect.adjusted(10, 0, -10, 0),
                         Qt.AlignHCenter | Qt.AlignVCenter, opt.text)

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        s.setHeight(max(s.height(), 38))
        return s


class MainWindow(QWidget):
    def __init__(self, video: str = None, demo: bool = False, record: str = None):
        super().__init__()
        self.setWindowTitle("智能健身教练")
        # 无框 + 半透明：圆角弧形窗口由 paintEvent 自绘
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(build_qss())

        # demo 录制时间基准: 帧序列(固定fps)与播报事件(monotonic)都相对 t0 对齐
        self._t0 = time.monotonic()
        self._audio_events = []           # AudioPlayer 实际播放事件 [{t, wav}]
        self._record_dir = None
        self._rec_frame = 0
        self._demo_mode = demo            # _build_ui 里弹层改 Tool 窗口用（Wayland grab 限制）

        W, H = self._screen_size()
        self.setFixedSize(W, H)
        self.audio = AudioPlayer(LANG, event_log=self._audio_events)
        set_analyzer_lang(LANG)   # 分析器文案与 UI 初始语言同步
        # 训练门闸: 待机(未开始) / 训练中 / 暂停 —— 未训练时识别只预览不记录
        self._started = False
        self._training = False
        self._build_ui(W, H)

        self.worker = Worker(video)
        self.worker.audio = self.audio
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_ready.connect(self.on_status)
        # 初始动作可用 FITNESS_EXERCISE 覆盖（板上验证各动作识别用）
        self.worker.set_exercise(os.environ.get("FITNESS_EXERCISE", "squat"))
        self._sync_combo()          # 初始下拉框选中项
        self._build_ref_popup()     # 弹层行依赖 worker.exercise（▶ 当前项指示）
        self.worker.start()
        QTimer.singleShot(1500, lambda: self.audio.play("welcome"))
        # 验证辅助: FITNESS_AUTOSTART=1 时 3s 后自动点开始训练（绕过待机门闸）
        if os.environ.get("FITNESS_AUTOSTART"):
            QTimer.singleShot(3000, self.btn_start.click)

        # 录屏帧录制: --record 指定目录（--demo 未指定时落到安装根 demo_out/）
        if record or demo:
            self._record_dir = record or str(
                Path(__file__).resolve().parent.parent / "demo_out")
            Path(self._record_dir).mkdir(parents=True, exist_ok=True)
            self._rec_timer = QTimer(self)
            self._rec_timer.timeout.connect(self._record_frame)
            self._rec_timer.start(1000 // RECORD_FPS)
            print(f"[record] -> {self._record_dir} @ {RECORD_FPS}fps", flush=True)

        # demo 自动预演: 分镜时间线驱动真实 UI 信号路径
        if demo:
            self._schedule_demo()

        # 调试: FITNESS_SNAPSHOT=/path.png 时 15s 后自动截图（验证 UI 用）
        #        FITNESS_POPUP=1 时截图前 1s 自动弹出下拉列表（验证弹层样式）
        snap = os.environ.get("FITNESS_SNAPSHOT")
        if snap:
            if os.environ.get("FITNESS_POPUP"):
                QTimer.singleShot(14000, self._open_ref_popup)
            QTimer.singleShot(15000, lambda: self._save_snapshot(snap))

    # ---------- demo 录制 ----------
    def _record_frame(self):
        # 弹层/播放器均为窗口子控件, self.grab() 直接包含
        pix = self.grab()
        if pix.isNull():
            return
        if pix.width() > 1280:       # 统一到 1280 宽，压缩单帧体积
            pix = pix.scaledToWidth(1280, Qt.SmoothTransformation)
        pix.save(str(Path(self._record_dir)
                     / f"frame_{self._rec_frame:05d}.jpg"), "JPG", 85)
        self._rec_frame += 1

    def _schedule_demo(self):
        import functools
        for ev in DEMO_SCRIPT:
            t, action = ev[0], ev[1]
            arg = ev[2] if len(ev) > 2 else None
            QTimer.singleShot(int(t * 1000),
                              functools.partial(self._demo_step, action, arg))
        print(f"[demo] scheduled {len(DEMO_SCRIPT)} events "
              f"(t0={time.strftime('%H:%M:%S')})", flush=True)

    def _demo_step(self, action: str, arg):
        print(f"[demo] {action} {arg or ''} "
              f"@ {time.monotonic() - self._t0:.1f}s", flush=True)
        if action in ("start", "pause", "resume"):
            self.btn_start.click()          # 三态门闸切换走真实点击路径
        elif action == "reset":
            self.btn_reset.click()
        elif action == "select":
            self.combo.setCurrentIndex(list(EXERCISES).index(arg))
        elif action == "popup":
            self._open_ref_popup()         # 展开动作列表（介绍六种动作+▶演示）
        elif action == "popdown":
            self.ref_popup.hide()
        elif action == "play_ref":
            self.play_ref(arg)             # 播放参考动作示范视频
        elif action == "close_ref":
            self.close_ref()
        elif action == "zoom_in":
            self.btn_zoom_in.click()        # 数字变焦＋（0.25x 步进）
        elif action == "zoom_reset":
            self.btn_zoom_reset.click()     # 变焦复位 1.0x
        elif action == "lang":
            self.btn_lang.click()
        elif action == "finish":
            self.finish()

    @staticmethod
    def _screen_size():
        """窗口尺寸自适应屏幕：四周统一留 ~4% 边距（同左右），
        钳制在 960x540 ~ 1360x860"""
        scr = QApplication.primaryScreen()
        if scr is None:
            return 1180, 600
        g = scr.availableGeometry()
        w = max(960, min(int(g.width() * 0.92), 1360))
        h = max(540, min(int(g.height() * 0.88), 860))
        return w, h

    def paintEvent(self, _):
        """自绘弧形圆角窗口: 渐变底 + 深色描边"""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0, QColor("#74C0FC"))
        grad.setColorAt(1.0, QColor("#B197FC"))
        p.setPen(QPen(QColor("#2D3436"), 4))
        p.setBrush(grad)
        p.drawRoundedRect(2, 2, w - 4, h - 4, 26, 26)

    def _save_snapshot(self, path):
        ok = self.grab().save(path)
        print(f"[snapshot] {path} {'ok' if ok else 'SAVE FAILED'}", flush=True)

    # ---------- UI ----------
    def _build_ui(self, W: int, H: int):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # 左: 视频预览（漫画分格）+ 变焦控制
        left = QVBoxLayout()
        left.setSpacing(8)
        panel_w = min(470, int(W * 0.40))

        self.video_label = QLabel(ut("cam_start"))
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        # 高度交给布局分配（Expanding 填满剩余空间），避免手工计算误差
        # 导致下方变焦按钮被窗口底边裁掉
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(320, 240)
        left.addWidget(self.video_label, 1)

        zoom_bar = QHBoxLayout()
        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_in = QPushButton("＋")
        self.btn_zoom_reset = QPushButton("1.0x")
        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_reset):
            b.setObjectName("zoom_btn")
            b.setFixedSize(56, 40)
        self.zoom_label = QLabel(ut("zoom", v=1.0))
        self.zoom_label.setObjectName("zoom_label")
        self.btn_zoom_in.clicked.connect(lambda: self.adjust_zoom(+0.25))
        self.btn_zoom_out.clicked.connect(lambda: self.adjust_zoom(-0.25))
        self.btn_zoom_reset.clicked.connect(lambda: self.adjust_zoom(0))
        zoom_bar.addWidget(self.btn_zoom_out)
        zoom_bar.addWidget(self.btn_zoom_in)
        zoom_bar.addWidget(self.btn_zoom_reset)
        zoom_bar.addStretch()
        zoom_bar.addWidget(self.zoom_label)
        left.addLayout(zoom_bar)
        root.addLayout(left)

        # 右: 控制面板（奶油底 + 粗描边）
        panel = QVBoxLayout()
        panel.setSpacing(10)
        panel.setContentsMargins(14, 14, 14, 14)

        # 标题行: 标题 + 语言切换按钮（右侧小按钮）
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self.title = QLabel(ut("title"))
        self.title.setObjectName("title")
        self.title.setAlignment(Qt.AlignCenter)
        self.btn_lang = QPushButton(ut("lang_btn"))
        self.btn_lang.setObjectName("btn_lang")
        self.btn_lang.setFixedSize(48, 36)
        self.btn_lang.setToolTip("中/EN")
        self.btn_lang.clicked.connect(self.toggle_lang)
        title_row.addWidget(self.title, 1)
        title_row.addWidget(self.btn_lang, 0, Qt.AlignTop)
        panel.addLayout(title_row)
        self.subtitle = QLabel(ut("subtitle"))
        self.subtitle.setObjectName("subtitle")
        self.subtitle.setAlignment(Qt.AlignCenter)
        panel.addWidget(self.subtitle)

        # 动作选择（下拉框：动作多了以后按钮铺满面板，改为单点展开选择）
        self.combo = QComboBox()
        for name in EXERCISES:
            self.combo.addItem(EXERCISE_NAMES[name][LANG], name)  # userData = 动作 key
        self.combo.setFixedHeight(52)
        self.combo.setMaxVisibleItems(len(EXERCISES))
        # 选中文字居中（默认左对齐，触屏上居中更醒目）
        self.combo.setEditable(True)
        self.combo.lineEdit().setReadOnly(True)
        self.combo.lineEdit().setAlignment(Qt.AlignCenter)
        self.combo.lineEdit().setStyleSheet(
            "background:transparent;border:none;color:white;font-weight:800;")
        # editable combo 的 lineEdit 会吞掉点击: 只点右侧箭头才弹窗。
        # 装事件过滤器让整个下拉框任意位置点击都能弹出（触屏友好）
        self.combo.lineEdit().installEventFilter(self)
        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        panel.addWidget(self.combo)
        # 动作列表弹层 + 参考视频播放器: 都做成主窗口的子控件。
        # 原因: Wayland 禁止客户端定位顶层弹窗（下拉列表会跑到屏幕中央），
        # 且 grabbing popup 需要真实输入序列（demo 程序化弹不出）；
        # 子控件在窗口坐标系内精确摆放, 录屏 grab() 也能直接抓到。
        self.ref_popup = QFrame(self)
        self.ref_popup.setObjectName("refPopup")
        self.ref_popup.hide()
        self.ref_player = QFrame(self)
        self.ref_player.setObjectName("refPlayer")
        self.ref_player.hide()
        self._build_ref_player()          # 弹层行在 worker 就绪后构建（见 __init__）
        QApplication.instance().installEventFilter(self)   # 弹层外点击关闭

        # 计数/计时（爆炸星形）+ 单位
        self.count_burst = BurstLabel()
        panel.addWidget(self.count_burst)
        self.unit_label = QLabel(ut("unit_reps"))
        self.unit_label.setObjectName("unit_label")
        self.unit_label.setAlignment(Qt.AlignCenter)
        panel.addWidget(self.unit_label)

        # 阶段 & 纠错 & FPS
        self.stage_label = QLabel(ut("ready"))
        self.stage_label.setObjectName("stage")
        self.stage_label.setAlignment(Qt.AlignCenter)
        self.stage_label.setMinimumHeight(34)
        self.problem_label = QLabel("")
        self.problem_label.setAlignment(Qt.AlignCenter)
        self.problem_label.setWordWrap(True)
        self.problem_label.setMinimumHeight(48)
        self.problem_label.setStyleSheet(PROBLEM_QSS["red"])
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("fps")
        self.fps_label.setAlignment(Qt.AlignCenter)
        panel.addWidget(self.stage_label)
        panel.addWidget(self.problem_label)
        panel.addWidget(self.fps_label)

        # 开始/暂停门闸（未开始训练前不计次，一切干扰只预览不记录）
        self.btn_start = QPushButton(ut("start_btn"))
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setFixedHeight(52)
        self.btn_start.clicked.connect(self.toggle_training)
        panel.addWidget(self.btn_start)

        # 结束按钮 + 归零按钮
        btn_row = QHBoxLayout()
        self.btn_reset = QPushButton(ut("reset_btn"))
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setFixedHeight(52)
        self.btn_reset.clicked.connect(self.reset_count)
        self.btn_finish = QPushButton(ut("finish_btn"))
        self.btn_finish.setObjectName("btn_finish")
        self.btn_finish.setFixedHeight(52)
        self.btn_finish.clicked.connect(self.finish)
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_finish)
        panel.addLayout(btn_row)

        panel.addStretch()
        frame = QFrame()
        frame.setObjectName("panel")
        frame.setLayout(panel)
        frame.setFixedWidth(panel_w)
        root.addWidget(frame)

    # ---------- 动作列表弹层 & 参考视频 ----------
    def _build_ref_popup(self):
        """动作列表弹层: 每行 = [动作名(点击切换)] [▶ 演示(播放参考视频)]。
        显示在 combo 正下方（窗口内坐标）；语言切换/选中变化时重建。"""
        self.ref_popup.setStyleSheet("""
            QFrame#refPopup { background:#FFF8E1; border:3px solid #2D3436;
                              border-radius:18px; }
            QPushButton#refItem { background:transparent; border:none; color:#2D3436;
                font-size:18px; font-weight:800; text-align:left;
                padding-left:14px; border-radius:12px; }
            QPushButton#refItem:hover { background:#FFE8A3; }
            QPushButton#refPlay { background:#51CF66; color:white; border:2px solid
                #2D3436; border-radius:12px; font-size:14px; font-weight:800; }
            QPushButton#refPlay:hover { background:#69DB7C; }""")
        self._ref_row_btns = []            # 本构建的按钮（重建时逐个删除）
        lay = self.ref_popup.layout()
        if lay is None:
            lay = QVBoxLayout(self.ref_popup)
        while lay.count():                      # 清空旧行（语言/选中态刷新）
            it = lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
            elif it.layout():                   # 行是 QHBoxLayout, 按钮在子布局里
                for i in range(it.layout().count()):
                    w = it.layout().itemAt(i).widget()
                    if w:
                        w.deleteLater()
                it.layout().deleteLater()
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)
        for key in EXERCISES:
            row = QHBoxLayout()
            cur = key == self.worker.exercise
            name_btn = QPushButton(("▶ " if cur else "") + EXERCISE_NAMES[key][LANG])
            name_btn.setObjectName("refItem")
            name_btn.setFixedHeight(46)
            name_btn.setCursor(Qt.PointingHandCursor)
            name_btn.clicked.connect(lambda _=False, k=key: self._pick_exercise(k))
            play_btn = QPushButton(ut("ref_btn"))
            play_btn.setObjectName("refPlay")
            play_btn.setFixedHeight(46)
            play_btn.setFixedWidth(92)
            play_btn.setCursor(Qt.PointingHandCursor)
            play_btn.clicked.connect(lambda _=False, k=key: self.play_ref(k))
            row.addWidget(name_btn, 1)
            row.addWidget(play_btn)
            lay.addLayout(row)
            self._ref_row_btns += [name_btn, play_btn]

    def _open_ref_popup(self):
        self._build_ref_popup()                  # 刷新 ▶ 指示与语言
        self.ref_popup.setFixedWidth(self.combo.width())   # 与下拉框同宽对齐
        self.ref_popup.adjustSize()
        pos = self.combo.mapTo(self, QPoint(0, self.combo.height() + 8))
        self.ref_popup.move(pos)
        self.ref_popup.raise_()
        self.ref_popup.show()
        print(f"[ref] popup geo {self.ref_popup.geometry()} "
              f"combo {self.combo.geometry()} win {self.width()}x{self.height()}",
              flush=True)
        for c in self.ref_popup.findChildren(QPushButton):
            print(f"[ref]   btn '{c.text()}' {c.geometry()} vis={c.isVisible()}",
                  flush=True)

    def _pick_exercise(self, key):
        self.ref_popup.hide()
        # 走 combo 真实信号路径（防误触冷却/回滚逻辑全部生效）
        self.combo.setCurrentIndex(list(EXERCISES).index(key))

    def _build_ref_player(self):
        """参考视频播放覆盖层: 盖在左侧视频区上, 播完自动收起, ✕ 关闭"""
        v = QVBoxLayout(self.ref_player)
        v.setContentsMargins(12, 8, 12, 12)
        v.setSpacing(6)
        top = QHBoxLayout()
        self.ref_title = QLabel("")
        self.ref_title.setObjectName("refTitle")
        close_btn = QPushButton("✕")
        close_btn.setObjectName("refClose")
        close_btn.setFixedSize(44, 38)
        close_btn.clicked.connect(self.close_ref)
        top.addWidget(self.ref_title, 1)
        top.addWidget(close_btn)
        self.ref_video = QLabel()
        self.ref_video.setAlignment(Qt.AlignCenter)
        v.addLayout(top)
        v.addWidget(self.ref_video, 1)
        self.ref_player.setStyleSheet("""
            QFrame#refPlayer { background:#1E1E2E; border:3px solid #2D3436;
                               border-radius:18px; }
            QLabel#refTitle { color:white; font-size:18px; font-weight:800; }
            QPushButton#refClose { background:#FF6B6B; color:white; border:2px
                solid #2D3436; border-radius:10px; font-size:16px; font-weight:800; }
            QPushButton#refClose:hover { background:#FF8787; }""")
        self._ref_cap = None
        self._ref_timer = QTimer(self)
        self._ref_timer.timeout.connect(self._ref_tick)

    def play_ref(self, key: str):
        path = str(Path(REF_DIR) / f"{key}.mp4")
        if not Path(path).exists():
            self.problem_label.setText(ut("ref_missing"))
            print(f"[ref] missing {path}", flush=True)
            return
        if self._ref_cap is not None:
            self._ref_cap.release()
        self._ref_cap = cv2.VideoCapture(path)
        self.ref_title.setText(ut("ref_playing",
                                  name=EXERCISE_NAMES[key][LANG]))
        g = self.video_label.geometry()          # 覆盖左侧视频区
        self.ref_player.setGeometry(g.adjusted(-6, -6, 6, 6))
        print(f"[ref] player geo {self.ref_player.geometry()} "
              f"video_label(raw) {g} "
              f"video abs {self.video_label.mapTo(self, QPoint(0,0))}",
              flush=True)
        self.ref_player.raise_()
        self.ref_player.show()
        self._ref_timer.start(33)                 # ~30fps
        print(f"[ref] play {key}", flush=True)

    def _ref_tick(self):
        if self._ref_cap is None:
            return
        ok, frame = self._ref_cap.read()
        if not ok:                                # 播完自动关闭（不循环）
            self.close_ref()
            return
        h, w = frame.shape[:2]
        s = min(self.ref_video.width() / w, self.ref_video.height() / h, 1.0)
        if s < 1.0:
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                     3 * rgb.shape[1], QImage.Format_RGB888)
        self.ref_video.setPixmap(QPixmap.fromImage(img))

    def close_ref(self):
        self._ref_timer.stop()
        self.ref_player.hide()
        if self._ref_cap is not None:
            self._ref_cap.release()
            self._ref_cap = None
        print("[ref] close", flush=True)

    # ---------- 回调 ----------
    def eventFilter(self, obj, e):
        # 只读 lineEdit 吞点击 -> 点文字区域也弹出动作列表
        if obj is self.combo.lineEdit() and e.type() == QEvent.MouseButtonRelease:
            self._open_ref_popup()
            return True
        # 弹层外任意点击关闭弹层（弹层是子控件, 自身点击不受影响）
        if (e.type() == QEvent.MouseButtonPress and self.ref_popup.isVisible()
                and obj is not None and obj is not self.ref_popup):
            w = obj.parentWidget() if hasattr(obj, "parentWidget") else None
            inside = False
            while w is not None:
                if w is self.ref_popup:
                    inside = True
                    break
                w = w.parentWidget()
            if not inside:
                self.ref_popup.hide()
        return super().eventFilter(obj, e)

    def adjust_zoom(self, delta: float):
        """delta=0 表示重置为 1.0x"""
        z = 1.0 if delta == 0 else self.worker.get_zoom() + delta
        self.worker.set_zoom(z)
        self.zoom_label.setText(ut("zoom", v=self.worker.get_zoom()))

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Escape:              # 无框窗口没有关闭钮，Esc 退出
            self.close()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.adjust_zoom(+0.25)
        elif k == Qt.Key_Minus:
            self.adjust_zoom(-0.25)
        elif k == Qt.Key_0:
            self.adjust_zoom(0)
        else:
            super().keyPressEvent(e)

    def toggle_lang(self):
        """中英切换: UI 文案 + 分析器文案 + 语音目录 三方同步"""
        global LANG
        LANG = "en" if LANG == "zh" else "zh"
        set_analyzer_lang(LANG)
        self.audio.set_lang(LANG)
        # 刷新全部静态文案
        self.title.setText(ut("title"))
        self.subtitle.setText(ut("subtitle"))
        self.btn_lang.setText(ut("lang_btn"))
        self.unit_label.setText(
            ut("unit_secs") if self.worker.exercise == "horse_stance"
            else ut("unit_reps"))
        self.btn_finish.setText(ut("finish_btn"))
        self.btn_start.setText(ut("pause_btn") if self._training
                               else (ut("resume_btn") if self._started
                                     else ut("start_btn")))
        self.zoom_label.setText(ut("zoom", v=self.worker.get_zoom()))
        # 下拉框条目重建（当前选中项保持）
        cur = self.worker.exercise
        self.combo.blockSignals(True)
        self.combo.clear()
        for name in EXERCISES:
            self.combo.addItem(EXERCISE_NAMES[name][LANG], name)
        self.combo.setCurrentIndex(list(EXERCISES).index(cur))
        self.combo.blockSignals(False)
        self._build_ref_popup()       # 弹层行文字跟随新语言
        # 重建分析器实例，让 stage 文案立即跟随新语言
        self.worker.set_exercise(cur)
        self.stage_label.setText(ut("ready"))
        # 播报确认切换（用当前动作名，直接体现新语言）
        self.audio.clear_queue()
        self.audio.play(EXERCISES[cur]["sound"])
        print(f"[lang] -> {LANG}", flush=True)

    def _on_combo_changed(self, idx: int):
        name = self.combo.itemData(idx)
        if name:
            self.switch_exercise(name, "combo")

    def switch_exercise(self, name: str, src: str = "code"):
        # 埋点: 记录每次切换的时间与真实来源
        print(f"[switch] -> {name} at {time.strftime('%H:%M:%S')} by {src}", flush=True)
        # 防误触1: 已是该模式则忽略（重复选当前项不重置计数、不重播语音）
        if name == self.worker.exercise:
            print(f"[switch] ignore: already in {name}", flush=True)
            self._sync_combo()
            return
        # 防误触2: 切换冷却 1.5s —— 运动时手掌扫过触摸屏会连碰多次
        now = time.monotonic()
        if now - getattr(self, "_last_switch_t", -1e9) < 1.5:
            print(f"[switch] ignore: cooldown", flush=True)
            self._sync_combo()
            return
        self._last_switch_t = now

        self.worker.set_exercise(name)
        self._sync_combo()
        self.count_burst.set_value(0, pulse=False)
        self.unit_label.setText(
            ut("unit_secs") if name == "horse_stance" else ut("unit_reps"))
        self.problem_label.setText("")
        # 清空待播队列 + 打断正在播的旧语音，避免"旧 start 播一半 -> 新模式名"的乱序听感
        self.audio.clear_queue()
        self.audio.play(EXERCISES[name]["sound"])
        if self._training:       # 训练中切换 = 立刻开练；待机/暂停只报模式名
            self.audio.play("start")

    def _sync_combo(self):
        """下拉框选中项与实际模式保持一致（拒绝切换时回滚显示）"""
        idx = list(EXERCISES).index(self.worker.exercise)
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)

    def on_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_status(self, count, stage, problems, fps):
        if not self._training:
            # 待机/暂停: 阶段栏显示门闸状态，不显示动作阶段与纠错
            stage = ut("paused") if self._started else ut("standby")
            problems = []
        self.count_burst.set_value(count)
        self.stage_label.setText(stage if stage else ut("ready"))
        if problems:
            shown = "、".join(UI[p][LANG] if p in UI else p for p in problems)
            self.problem_label.setText(ut("problem", p=shown))
            self.problem_label.setStyleSheet(PROBLEM_QSS["red"])
        elif not self._training or stage in ("", UI["no_person"][LANG]):
            self.problem_label.setText("")
        else:
            self.problem_label.setText(ut("good_job"))
            self.problem_label.setStyleSheet(PROBLEM_QSS["green"])
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self._fps_log = getattr(self, "_fps_log", 0) + 1
        if self._fps_log % 50 == 0:   # 定期把 fps 打到日志便于远程观察
            print(f"[status] count={count} stage={stage} fps={fps:.1f}", flush=True)

    def toggle_training(self):
        """训练门闸: 待机→开始 / 训练中→暂停 / 暂停→继续。
        未开始或暂停时，识别只做预览，任何动作/干扰都不计入记录。"""
        if not self._started:                    # 待机 -> 开始训练
            self._started = True
            self._training = True
            self.worker.set_training(True)
            self.audio.clear_queue()
            self.audio.play("start")
            print("[session] training start", flush=True)
        elif self._training:                     # 训练中 -> 暂停
            self._training = False
            self.worker.set_training(False)
            print(f"[session] paused at count={self.worker.count}", flush=True)
        else:                                    # 暂停 -> 继续
            self._training = True
            self.worker.set_training(True)
            print("[session] training resumed", flush=True)
        # 刷新按钮文案/配色（属性选择器需重新 polish 才生效）
        self.btn_start.setText(ut("pause_btn") if self._training
                               else (ut("resume_btn") if self._started
                                     else ut("start_btn")))
        self.btn_start.setProperty("mode", "pause" if self._training else "")
        self.btn_start.style().unpolish(self.btn_start)
        self.btn_start.style().polish(self.btn_start)

    def reset_count(self):
        """归零：重置当前动作的计数/计时"""
        self.worker.reset_analyzer()
        self.count_burst.set_value(0, pulse=False)
        self.problem_label.setText("")
        print("[reset] count reset to 0", flush=True)

    def finish(self):
        self.audio.play("finish")
        QTimer.singleShot(2500, self.close)   # close 统一走 closeEvent（写 meta 后随 lastWindowClosed 退出）

    def closeEvent(self, e):
        self.worker.stop()
        if getattr(self, "_rec_timer", None):
            self._rec_timer.stop()
        self.audio.shutdown()                 # join 音频线程，事件日志写完
        # 录制元数据: 合成脚本按 (event.t - 0) * record_fps 定位帧、按 t 混音
        if self._record_dir is not None:
            meta = {
                "record_fps": RECORD_FPS,
                "frames": self._rec_frame,
                "t0_wall": time.strftime("%Y-%m-%d %H:%M:%S"),
                "audio_events": [
                    {"t": round(ev["t"] - self._t0, 3), "wav": ev["wav"]}
                    for ev in self._audio_events],
            }
            (Path(self._record_dir) / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1))
            print(f"[record] done: {self._rec_frame} frames, "
                  f"{len(meta['audio_events'])} audio events", flush=True)
        super().closeEvent(e)


def main():
    ap = argparse.ArgumentParser(description="智能健身教练")
    ap.add_argument("--video", help="用视频文件代替 USB 摄像头（调试用）")
    ap.add_argument("--demo", action="store_true",
                    help="自动预演: 按分镜时间线自动操作 UI（USB 摄像头画面做背景）")
    ap.add_argument("--record", metavar="DIR",
                    help="录屏帧输出目录（--demo 未指定时默认安装根 demo_out/）")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    global FONT_FAMILY
    fam = load_cjk_font()
    FONT_FAMILY = fam
    font = QFont(fam, 11)
    app.setFont(font)
    print(f"[font] app font family: {fam}", flush=True)
    print(f"[lang] initial={LANG}", flush=True)
    win = MainWindow(video=args.video, demo=args.demo, record=args.record)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
