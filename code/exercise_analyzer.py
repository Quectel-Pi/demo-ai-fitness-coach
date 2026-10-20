#!/usr/bin/env python3
"""深蹲 / 开合跳 / 高抬腿 / 弓步蹲 / 侧平举 / 马步蹲（计时）分析器。

输入: PoseEstimator 输出的 (17,3) 关键点（原图坐标，y 向下为正）
输出: 每帧调用 update()，返回 Feedback(事件) —— 计数、纠错提示、当前阶段

文案国际化: stage/problems 均输出当前语言文本（set_lang 切换 zh/en），
事件名与音频文件名保持语言无关。
"""
from dataclasses import dataclass, field
from enum import Enum
import time as _time
from typing import Optional

import numpy as np

from pose_estimator import KP

# ---------------- i18n: 分析器文案 ----------------
_STRINGS = {
    "no_person":    {"zh": "未检测到人", "en": "No person detected"},
    "knee_angle":   {"zh": "膝角", "en": "knee"},
    "hip_angle":    {"zh": "髋角", "en": "hip"},
    # 深蹲
    "squat_down":   {"zh": "下蹲中 ({a} {deg:.0f}°)", "en": "Squatting ({a} {deg:.0f}°)"},
    "squat_stand":  {"zh": "站立", "en": "Standing"},
    "squat_reject": {"zh": "上次蹲不够深({deg:.0f}°) 未计次",
                     "en": "Last squat too shallow ({deg:.0f}°) — not counted"},
    "squat_bend":   {"zh": "弯腰了，请蹲下而非弯腰", "en": "Bending over — squat down, don't bend"},
    "knees_out":    {"zh": "膝盖出镜了，后退一点", "en": "Knees out of frame — step back"},
    "squat_down_h": {"zh": "下蹲中 (髋高估算)", "en": "Squatting (hip-height est.)"},
    "wrists_out":   {"zh": "手腕出镜了", "en": "Wrists out of frame"},
    "jj_full_body": {"zh": "开合跳需全身入镜，后退一点",
                     "en": "Jumping jacks need full body — step back"},
    # 开合跳
    "jj_open":      {"zh": "打开", "en": "Open"},
    "jj_closed":    {"zh": "并拢", "en": "Closed"},
    "jj_ready":     {"zh": "并拢就位", "en": "Ready"},
    "jj_ratio":     {"zh": "{state} (脚距/肩宽 {r:.2f})",
                     "en": "{state} (feet/shoulder {r:.2f})"},
    # 高抬腿
    "hk_up":        {"zh": "抬腿中 (膝高 {r:.2f})", "en": "Kicking (knee {r:.2f})"},
    "hk_down":      {"zh": "双腿着地 (膝高 {r:.2f})", "en": "Feet down (knee {r:.2f})"},
    # 弓步蹲
    "side_l":       {"zh": "左", "en": "left"},
    "side_r":       {"zh": "右", "en": "right"},
    "lunge_down":   {"zh": "弓步中 ({side}腿 {a} {deg:.0f}°)",
                     "en": "Lunging ({side} leg, {a} {deg:.0f}°)"},
    "lunge_stand":  {"zh": "站立", "en": "Standing"},
    "lunge_reject": {"zh": "上次弓步不够低({deg:.0f}°) 未计次",
                     "en": "Last lunge too high ({deg:.0f}°) — not counted"},
    # 侧平举
    "raise_up":     {"zh": "双臂举起", "en": "Arms raised"},
    "raise_down":   {"zh": "双臂下垂", "en": "Arms lowered"},
    # 马步蹲
    "horse_hold":   {"zh": "坚持中 {n} 秒 ({a} {deg:.0f}°)",
                     "en": "Holding {n}s ({a} {deg:.0f}°)"},
    "horse_rest":   {"zh": "休息 (已坚持 {n} 秒)", "en": "Resting ({n}s held)"},
    "horse_ready":  {"zh": "站直准备", "en": "Stand ready"},
    "horse_wide":   {"zh": "双脚再分开一点（马步要宽距）",
                     "en": "Widen your stance (horse stance)"},
    "horse_hold_h": {"zh": "坚持中 {n} 秒 (髋高估算)",
                     "en": "Holding {n}s (hip-height est.)"},
    # problems（UI 展示的问题关键词）
    "knee_in":      {"zh": "膝内扣", "en": "Knees caving in"},
    "trunk_lean":   {"zh": "背部前倾", "en": "Leaning forward"},
    "jj_arms_p":    {"zh": "手臂未过顶", "en": "Arms not overhead"},
    "jj_feet_p":    {"zh": "双脚分不够开", "en": "Feet not wide enough"},
}
_LANG = "zh"


def set_lang(lang: str):
    """切换分析器输出文案语言（zh/en），UI 与语音同步切换"""
    global _LANG
    _LANG = lang


def T(key: str, **kw) -> str:
    """取当前语言的文案模板并填充参数"""
    return _STRINGS[key][_LANG].format(**kw)


def _angle(a, b, c) -> float:
    """三点夹角（b 为顶点），单位度"""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _mid(p1, p2):
    return (p1 + p2) / 2.0


@dataclass
class Feedback:
    event: Optional[str] = None      # "count" / 提示音频名 / None
    count: int = 0                   # 最新计数（event=count 时有效）
    stage: str = ""                  # 当前阶段描述
    problems: list = field(default_factory=list)  # 本帧检测到的问题（UI 展示）


# 关键点置信度阈值：实测 yolo11n 在 640x480 远距下 kpt conf 常在 0.1~0.4
KPT_CONF_THRES = 0.25
# 测量关键点（膝/踝/腕）置信度阈值：比躯干基准更高，防低 conf 噪声位置误判
KPT_MEASURE_THRES = 0.30
# 高抬腿专用：动作幅度大关键点更容易丢，单独放宽
HK_KPT_CONF_THRES = 0.15


def _measure_ok(kpt, y_lo: float, y_hi: float,
                conf: float = KPT_MEASURE_THRES) -> bool:
    """测量关键点可用性: 置信度够 + y 坐标在解剖学合理范围 [y_lo, y_hi] 内。
    防止低置信度噪声位置（如膝高 -0.38 = 膝盖在髋上方）乱触发计次。"""
    return kpt[2] >= conf and y_lo <= kpt[1] <= y_hi


class SquatAnalyzer:
    """深蹲：膝盖角度状态机 + 三类纠错（深度不足/膝内扣/背部前倾）

    脚踝不可见时（取景只到膝盖/大腿），自动降级为 髋角(肩-髋-膝) 状态机，
    保证上半身入镜即可计次。"""

    # 阈值（度）—— 膝角模式（已放宽：体育级标准过严，娱乐计次友好优先）
    STAND = 145        # 站立（原155：起身不完全时也允许计次）
    DEEP = 115         # 蹲到位（原105：大腿平行位约100-110°，日常蹲很难低于105）
    SHALLOW = 130      # 下蹲但不够深 -> 提示
    TRUNK_LEAN = 55    # 躯干前倾阈值
    KNEE_IN = 0.09     # 膝内扣: 膝盖x 相对髋-踝连线的水平偏移比例

    # 阈值（度）—— 髋角兜底模式（肩-髋-膝；站立≈170，平行蹲≈90~110）
    HIP_STAND = 140
    HIP_DEEP = 115
    HIP_ENTRY = 125

    BEND_GATE = 55       # 弯腰门闸: 躯干前倾 >55° = 弯腰不是蹲
    CONFIRM_DOWN = 3     # 连续 N 帧低于入闸阈值才确认下蹲（防抖动误入）
    CONFIRM_UP = 2       # 连续 N 帧高于出闸阈值才确认起身（防抖动误出）

    def __init__(self):
        self.count = 0
        self.state = "stand"          # stand / down
        self.min_knee = 180.0         # 本次下蹲的最低膝角
        self.lean_at_bottom = 0.0
        self.knee_in_frames = 0
        self._reject_note = ""        # 上次未计次原因（UI 展示 2s）
        self._reject_until = 0.0
        # 高度兜底模式（膝盖出镜时用）
        self._hip_base = None         # 站立时髋部 y 基线（EMA）
        self._torso_ref = None        # 站立时躯干长参考（EMA）
        self._down_via = "angle"      # 本次下蹲来源: angle / height
        self._down_t = 0.0            # 进入 down 的时刻（低位滞留重置用）
        # 帧确认计数器（防单帧抖动触发状态切换）
        self._down_votes = 0          # stand 时: 连续低于入闸阈值的帧数
        self._up_votes = 0            # down 时: 连续高于出闸阈值的帧数
        # 连续上半身可见帧数：高度兜底模式要求连续可见才判定，防单帧闪现误计
        self._upper_visible = 0

    def update(self, k: np.ndarray) -> Feedback:
        if k is None:
            # 未检测到人也要回传当前计数，否则 UI 计数会在 0↔N 之间闪烁
            return Feedback(stage=T("no_person"), count=self.count)

        lh, rh = k[KP["left_hip"]], k[KP["right_hip"]]
        lk, rk = k[KP["left_knee"]], k[KP["right_knee"]]
        la, ra = k[KP["left_ankle"]], k[KP["right_ankle"]]
        ls, rs = k[KP["left_shoulder"]], k[KP["right_shoulder"]]

        # 上半身（肩+髋）可见 -> 即使膝盖出镜也还有救（高度兜底）
        upper_ok = (ls[2] >= KPT_CONF_THRES and rs[2] >= KPT_CONF_THRES and
                    lh[2] >= KPT_CONF_THRES and rh[2] >= KPT_CONF_THRES)
        if not upper_ok:
            self._upper_visible = 0   # 上半身丢失，重置连续计数
            return Feedback(stage=T("no_person"), count=self.count)
        self._upper_visible += 1

        hip_mid = _mid(lh[:2], rh[:2])
        shoulder_mid = _mid(ls[:2], rs[:2])
        torso = float(np.linalg.norm(shoulder_mid - hip_mid))
        knee_conf_ok = lk[2] >= KPT_CONF_THRES and rk[2] >= KPT_CONF_THRES
        # 躯干前倾: 以肩中点为顶点，"肩->髋"向量与竖直向下的夹角（站立=0°，前倾越大越大）
        trunk = _angle(hip_mid, shoulder_mid, shoulder_mid + np.array([0.0, 1.0]))
        now = _time.monotonic()

        # 站立时校准髋部高度基线 + 躯干长（EMA）—— 膝盖出镜时靠它计次
        if self.state == "stand" and torso >= 25:
            if self._hip_base is None:
                self._hip_base, self._torso_ref = float(hip_mid[1]), torso
            else:
                d = float(hip_mid[1]) - self._hip_base
                if d < -0.5 * self._torso_ref:
                    # 基线偏低（启动时人在低位）-> 直接重锚
                    self._hip_base, self._torso_ref = float(hip_mid[1]), torso
                    print(f"[squat] re-anchor base={self._hip_base:.0f} torso_ref={self._torso_ref:.0f}", flush=True)
                elif abs(d) < 0.3 * self._torso_ref:
                    # 仅在"确实站着"时校准 —— 防止下蹲过程中基线追着髋部跑
                    self._hip_base += 0.15 * (float(hip_mid[1]) - self._hip_base)
                    self._torso_ref += 0.15 * (torso - self._torso_ref)

        # ---- 膝盖出镜: 高度兜底模式（肩髋可见，用髋部下沉量计次）----
        # 要求连续 2 帧上半身可见才进入兜底模式，防单帧闪现+基线漂移误计
        if not knee_conf_ok:
            if self._upper_visible < 2:
                return Feedback(stage=T("no_person"), count=self.count)
            return self._update_height(float(hip_mid[1]), trunk, now)

        # ---- 膝盖可见: 角度模式（原逻辑）----
        # 两侧脚踝均可见 -> 膝角模式；否则髋角兜底（脚出镜仍可计次）
        ankles_ok = la[2] >= KPT_CONF_THRES and ra[2] >= KPT_CONF_THRES
        if ankles_ok:
            knee = (_angle(lh[:2], lk[:2], la[:2]) +
                    _angle(rh[:2], rk[:2], ra[:2])) / 2
            t_stand, t_deep, t_entry = self.STAND, self.DEEP, self.SHALLOW - 5
            ang_name = T("knee_angle")
        else:
            if ls[2] < KPT_CONF_THRES or rs[2] < KPT_CONF_THRES:
                return Feedback(stage=T("no_person"), count=self.count)
            knee = (_angle(ls[:2], lh[:2], lk[:2]) +
                    _angle(rs[:2], rh[:2], rk[:2])) / 2
            t_stand, t_deep, t_entry = self.HIP_STAND, self.HIP_DEEP, self.HIP_ENTRY
            ang_name = T("hip_angle")

        # 膝内扣：膝盖相对 髋-踝 连线朝"身体中线方向"的水平偏移。
        # 左右分开判：内扣时两侧都朝中线偏（有符号一正一负），直接平均会抵消；
        # 取两侧内偏量的较大者（外偏一侧计 0），任一膝内扣超阈值即累计。
        def knee_in_offset(hip, knee_pt, ankle, mid_x):
            t = (knee_pt[1] - hip[1]) / max(ankle[1] - hip[1], 1e-6)
            line_x = hip[0] + (ankle[0] - hip[0]) * np.clip(t, 0, 1)
            off = (knee_pt[0] - line_x) / max(abs(ankle[0] - hip[0]), 30)
            # 朝中线偏移 = 内扣；背离中线 = 外偏（计 0）
            toward_mid = (mid_x - hip[0])
            inward = off if (off > 0) == (toward_mid > 0) else 0.0
            return inward

        if ankles_ok:
            mid_x = (lh[0] + rh[0]) / 2.0
            inL = knee_in_offset(lh, lk, la, mid_x)
            inR = knee_in_offset(rh, rk, ra, mid_x)
            off = max(inL, inR)          # 任一膝内扣即算
            if off > self.KNEE_IN:
                self.knee_in_frames += 1
            else:
                self.knee_in_frames = max(0, self.knee_in_frames - 1)

        problems = []
        event = None

        # 弯腰门闸: 躯干前倾过大时髋角同样会变小，但那是弯腰不是蹲。
        # 真蹲下时躯干前倾约 20~40°，弯腰时 >55°。用 55° 作门闸。

        if self.state == "stand":
            if knee < t_entry and trunk < self.BEND_GATE:
                self._down_votes += 1
                self._up_votes = 0
                if self._down_votes >= self.CONFIRM_DOWN:
                    self.state = "down"
                    self._down_via = "angle"
                    self._down_t = now
                    self.min_knee = knee
                    self.lean_at_bottom = trunk
                    self._down_votes = 0
                    print(f"[squat] down via=angle knee={knee:.0f} trunk={trunk:.0f}", flush=True)
            else:
                self._down_votes = 0
                if knee < t_entry and trunk >= self.BEND_GATE:
                    # 弯腰导致的髋角变小 -> 不计次，提示用户
                    self._reject_note = T("squat_bend")
                    self._reject_until = now + 2.0
        else:  # down
            self.min_knee = min(self.min_knee, knee)
            self.lean_at_bottom = max(self.lean_at_bottom, trunk)
            if knee > t_stand:
                self._up_votes += 1
                if self._up_votes >= self.CONFIRM_UP:
                    # 起身完成，判定本次深蹲质量
                    if self._down_via == "height" or self.min_knee < t_deep:
                        self.count += 1
                        event = "count"
                        print(f"[squat] count={self.count} via={self._down_via} min_knee={self.min_knee:.0f}", flush=True)
                    else:
                        event = "squat_depth"      # 蹲得不够深，不计次
                        self._reject_note = T("squat_reject", deg=self.min_knee)
                        self._reject_until = _time.monotonic() + 2.0
                        print(f"[squat] reject depth min_knee={self.min_knee:.0f}", flush=True)
                    if self.knee_in_frames > 6:
                        problems.append("knee_in")
                    if self.lean_at_bottom > self.TRUNK_LEAN:
                        problems.append("trunk_lean")
                    self.state = "stand"
                    self.min_knee = 180.0
                    self.knee_in_frames = 0
                    self._up_votes = 0
            else:
                self._up_votes = 0

        if self.state == "down":
            if self.knee_in_frames > 6:
                problems.append("knee_in")
            if trunk > self.TRUNK_LEAN:
                problems.append("trunk_lean")
            stage = T("squat_down", a=ang_name, deg=knee)
        elif _time.monotonic() < self._reject_until:
            stage = self._reject_note
        else:
            stage = T("squat_stand")

        # 实时纠错播报优先级：膝内扣 > 背部前倾
        if event is None and self.state == "down":
            if self.knee_in_frames == 7:
                event = "squat_knee"
            elif trunk > self.TRUNK_LEAN + 8:
                event = "squat_back"

        return Feedback(event=event, count=self.count, stage=stage, problems=problems)

    # 入闸/出闸阈值（躯干长倍数）：深蹲髋部下沉约 0.7~0.85 倍躯干长，
    # 入闸 0.65 ≈ 真正蹲低的门槛（0.55 太松导致弯腰/移位误触发），
    # 出闸 0.22 提供迟滞防抖
    H_ENTER = 0.65
    H_EXIT = 0.22

    def _update_height(self, hip_y: float, trunk: float, now: float) -> Feedback:
        """高度兜底模式：膝盖出镜（下半身出画面）时，用髋部相对站立基线的下沉量计次。

        肩/髋可见即可工作；站立基线在 update() 里持续 EMA 校准。
        入闸/出闸均需连续帧确认，防单帧抖动误触发。
        注: 髋部高度无法区分"深蹲"和"蹲低看屏幕"，误计风险用户已接受。"""
        problems = []
        if self._hip_base is None or self._torso_ref is None:
            # 基线未就绪（刚启动/人太远）——只提示，不计次
            return Feedback(stage=T("knees_out"), count=self.count)
        drop = hip_y - self._hip_base
        t_enter, t_exit = self.H_ENTER * self._torso_ref, self.H_EXIT * self._torso_ref

        if self.state == "stand":
            if drop > t_enter and trunk < self.BEND_GATE:
                self._down_votes += 1
                self._up_votes = 0
                if self._down_votes >= self.CONFIRM_DOWN:
                    self.state = "down"
                    self._down_via = "height"
                    self._down_t = now
                    self.min_knee = 180.0
                    self.lean_at_bottom = trunk
                    self._down_votes = 0
                    stage = T("squat_down_h")
                    print(f"[squat] down via=height drop={drop:.0f} t_enter={t_enter:.0f} "
                          f"trunk={trunk:.0f} base={self._hip_base:.0f} torso_ref={self._torso_ref:.0f}", flush=True)
                else:
                    stage = self._reject_note if now < self._reject_until else T("knees_out")
            elif drop > t_enter and trunk >= self.BEND_GATE:
                self._down_votes = 0
                self._reject_note = T("squat_bend")
                self._reject_until = now + 2.0
                stage = self._reject_note
            else:
                self._down_votes = 0
                stage = self._reject_note if now < self._reject_until else T("knees_out")
        else:  # down
            self.lean_at_bottom = max(self.lean_at_bottom, trunk)
            if drop < t_exit:
                self._up_votes += 1
                if self._up_votes >= self.CONFIRM_UP:
                    # 起身完成
                    event = None
                    if self._down_via == "angle" and self.min_knee >= self.DEEP:
                        self._reject_note = T("squat_reject", deg=self.min_knee)
                        self._reject_until = now + 2.0
                        stage = self._reject_note
                    else:
                        self.count += 1
                        event = "count"
                        stage = T("squat_stand")
                        print(f"[squat] count={self.count} via=height drop={drop:.0f} t_exit={t_exit:.0f}", flush=True)
                    if self.lean_at_bottom > self.TRUNK_LEAN:
                        problems.append("trunk_lean")
                    self.state = "stand"
                    self.min_knee = 180.0
                    self.knee_in_frames = 0
                    self._up_votes = 0
                    return Feedback(event=event, count=self.count, stage=stage, problems=problems)
                else:
                    if trunk > self.TRUNK_LEAN:
                        problems.append("trunk_lean")
                    stage = T("squat_down_h")
            elif now - self._down_t > 4.0:
                # 低位滞留超时 = 蹲低看屏幕/系鞋带/调整站位，不是深蹲 —— 重置基线防卡死
                self._hip_base = hip_y
                self.state = "stand"
                self.min_knee = 180.0
                self._up_votes = 0
                stage = T("knees_out")
                print(f"[squat] dwell-reset drop={drop:.0f}", flush=True)
            else:
                self._up_votes = 0
                if trunk > self.TRUNK_LEAN:
                    problems.append("trunk_lean")
                stage = T("squat_down_h")
        return Feedback(count=self.count, stage=stage, problems=problems)


class JumpingJackAnalyzer:
    """开合跳：手脚开合状态机 + 纠错（手臂未过顶/双脚分不够开）"""

    FOOT_OPEN = 1.35     # 脚距 > 1.35 肩宽 = 打开
    FOOT_CLOSE = 0.6     # 脚距 < 0.6 肩宽 = 并拢
    CONFIRM = 3          # 连续 N 帧确认状态切换
    READY_CONFIRM = 3    # 启动就位: 连续 N 帧观察到并拢才开始计次
                          # （防启动时已张开双臂双脚被直接记 1 次）

    def __init__(self):
        self.count = 0
        self.state = "ready"        # ready 等待并拢就位 / closed / open
        self._ready_votes = 0
        self._open_votes = 0
        self._close_votes = 0
        self.warned_arms = False
        self.warned_feet = False

    def update(self, k: np.ndarray) -> Feedback:
        # 必需: 双肩（宽度基准）+ 双踝（脚距判据）；鼻/腕/肘可选
        if k is None or any(k[KP[n]][2] < KPT_CONF_THRES
                            for n in ("left_shoulder", "right_shoulder")):
            self._ready_votes = 0
            self._open_votes = 0
            self._close_votes = 0
            return Feedback(stage=T("no_person"), count=self.count)
        ls, rs = k[KP["left_shoulder"]], k[KP["right_shoulder"]]
        la, ra = k[KP["left_ankle"]], k[KP["right_ankle"]]
        shoulder_mid_y = (ls[1] + rs[1]) / 2.0
        torso_ref = 200.0   # 兜底躯干长（无髋数据时用）
        lh, rh = k[KP["left_hip"]], k[KP["right_hip"]]
        if lh[2] >= KPT_CONF_THRES and rh[2] >= KPT_CONF_THRES:
            torso_ref = max((lh[1] + rh[1]) / 2.0 - shoulder_mid_y, 30.0)
        y_lo = shoulder_mid_y - torso_ref
        y_hi = shoulder_mid_y + 3.0 * torso_ref
        if not _measure_ok(la, y_lo, y_hi) or not _measure_ok(ra, y_lo, y_hi):
            # 踝出镜/噪声: 脚距判据没了 -> 开合跳没法判，清票并提示全身入镜
            self._open_votes = 0
            self._close_votes = 0
            self._ready_votes = 0
            return Feedback(stage=T("jj_full_body"), count=self.count)

        lw, rw = k[KP["left_wrist"]], k[KP["right_wrist"]]
        le, re = k[KP["left_elbow"]], k[KP["right_elbow"]]
        nose = k[KP["nose"]]

        # 手臂过顶判定: 腕可见 -> 腕高于鼻(鼻不可见则高于肩);
        # 腕出镜(手举过头顶出画面顶部) -> 降级用肘: 肘高于肩 ≈ 手臂过顶
        w_y_lo = shoulder_mid_y - 1.5 * torso_ref
        w_y_hi = shoulder_mid_y + 0.8 * torso_ref
        if _measure_ok(lw, w_y_lo, w_y_hi) and _measure_ok(rw, w_y_lo, w_y_hi):
            ref_y = nose[1] if nose[2] >= KPT_CONF_THRES else shoulder_mid_y
            arms_up = lw[1] < ref_y and rw[1] < ref_y
        elif le[2] >= KPT_CONF_THRES and re[2] >= KPT_CONF_THRES:
            arms_up = le[1] < shoulder_mid_y and re[1] < shoulder_mid_y
        else:
            # 腕+肘均不可见: 手臂状态未知，清票并提示
            self._open_votes = 0
            self._close_votes = 0
            self._ready_votes = 0
            return Feedback(stage=T("wrists_out"), count=self.count)

        shoulder_w = abs(rs[0] - ls[0]) + 30          # +30 兜底侧身时肩宽过小
        ankle_w = abs(ra[0] - la[0])

        problems = []
        event = None
        fully_open = ankle_w > shoulder_w * self.FOOT_OPEN and arms_up

        if self.state == "ready":
            # 就位确认: 先观察到并拢（含手放下），才开始正常计次
            closed_ready = (ankle_w < shoulder_w * self.FOOT_CLOSE and not arms_up)
            self._ready_votes = self._ready_votes + 1 if closed_ready else 0
            if self._ready_votes >= self.READY_CONFIRM:
                self.state = "closed"
                self._ready_votes = 0
        elif self.state == "closed":
            self._open_votes = self._open_votes + 1 if fully_open else 0
            if self._open_votes >= self.CONFIRM:
                self.state = "open"
                self.count += 1        # 一次完整"打开"计 1 次
                event = "count"
                self._open_votes = 0
                self.warned_arms = False
                self.warned_feet = False
        else:  # open
            closed = ankle_w < shoulder_w * self.FOOT_CLOSE
            self._close_votes = self._close_votes + 1 if closed else 0
            if self._close_votes >= self.CONFIRM:
                self.state = "closed"
                self._close_votes = 0
            # 纠错（打开状态下评估，一次性提示）
            if not arms_up and not self.warned_arms:
                problems.append("jj_arms_p")
                event = "jj_arms"
                self.warned_arms = True
            if ankle_w < shoulder_w * 1.1 and not self.warned_feet and self._close_votes == 0:
                problems.append("jj_feet_p")
                if event is None:
                    event = "jj_feet"
                self.warned_feet = True

        if self.state == "ready":
            # 就位阶段提示：并拢手脚准备
            state = T("jj_ready")
        else:
            state = T("jj_open") if self.state == "open" else T("jj_closed")
        stage = T("jj_ratio", state=state, r=ankle_w / shoulder_w)
        return Feedback(event=event, count=self.count, stage=stage, problems=problems)


class HighKneesAnalyzer:
    """高抬腿：任一膝抬至髋部高度 -> 落地还原 -> 计 1 次（左右腿独立计次）。

    左右腿分别维护 down/up 状态: 抬起确认后计 1 次，该腿落地还原后才准备
    下一次 —— 交替抬腿（左、右、左、右）各计 1 次，不会因"始终有一腿
    满足抬起"而漏计。
    膝盖不可见（无法判断）时清票不做证据，防丢帧把同一次抬腿拆成 2 次。"""

    LIFT_TOL = 0.50     # 膝在髋下方该躯干长比例以内 = 抬到位（办公室慢速版）
    ANKLE_DIFF = 0.08   # 抬起侧脚踝需高于另一侧的躯干长比例
    LOW_FRAMES = 8      # 「抬得不够高」持续帧数 -> 纠错提示
    CONFIRM_UP = 2      # 连续 N 帧膝到位才确认抬腿（防走路/晃腿误触发）
    CONFIRM_DOWN = 2    # 连续 N 帧落地才确认还原（防抖动提前计次）

    def __init__(self):
        self.count = 0
        # 左右腿独立状态: "down"着地待抬 / "up"已抬起待还原
        self.leg_state = {"L": "down", "R": "down"}
        self._up_votes = {"L": 0, "R": 0}
        self._down_votes = {"L": 0, "R": 0}
        self._low_frames = 0

    def update(self, k: np.ndarray) -> Feedback:
        # 必需: 双肩 + 双髋（躯干基准，0.15 阈值够用）
        thres = HK_KPT_CONF_THRES
        if k is None or any(k[KP[n]][2] < thres
                            for n in ("left_shoulder", "right_shoulder",
                                      "left_hip", "right_hip")):
            return Feedback(stage=T("no_person"), count=self.count)

        ls, rs = k[KP["left_shoulder"]], k[KP["right_shoulder"]]
        lh, rh = k[KP["left_hip"]], k[KP["right_hip"]]
        lk, rk = k[KP["left_knee"]], k[KP["right_knee"]]
        la, ra = k[KP["left_ankle"]], k[KP["right_ankle"]]

        torso = max(_mid(lh[:2], rh[:2])[1] - _mid(ls[:2], rs[:2])[1], 30.0)
        hip_y = _mid(lh[:2], rh[:2])[1]
        ankles_ok = la[2] >= thres and ra[2] >= thres

        # 逐腿判定: 膝盖 conf >= 0.30 且位置合理才采信；不可见返回 None（无法判断）
        def lifted(knee_pt, ankle, other_ankle):
            if not _measure_ok(knee_pt, ls[1], hip_y + 1.5 * torso):
                return None
            high = knee_pt[1] < hip_y + self.LIFT_TOL * torso
            if high and ankles_ok and ankle[2] >= thres and other_ankle[2] >= thres:
                high = ankle[1] < other_ankle[1] - self.ANKLE_DIFF * torso
            # 注意: numpy 标量比较产生 np.bool_，`np.bool_(False) is False` 为
            # False，会绕过状态机的 is False / is None 分支 -> 必须转 Python bool
            return bool(high)

        l_lifted = lifted(lk, la, ra)
        r_lifted = lifted(rk, ra, la)
        # d: 取可见膝中较高的那个；都不可见则用躯干长占位（不会触发任何阈值）
        visible_knees = [kk[1] for kk in (lk, rk)
                         if _measure_ok(kk, ls[1], hip_y + 1.5 * torso)]
        d = min(visible_knees) - hip_y if visible_knees else torso

        event = None
        for side, is_up in (("L", l_lifted), ("R", r_lifted)):
            if self.leg_state[side] == "down":
                if is_up:      # True 才是抬起证据
                    self._up_votes[side] += 1
                    if self._up_votes[side] >= self.CONFIRM_UP:
                        self.leg_state[side] = "up"
                        self.count += 1
                        event = "count"
                        self._low_frames = 0
                elif is_up is None:
                    # 膝盖不可见 = 无法判断，清票不做证据（防跨丢帧拼凑计次）
                    self._up_votes[side] = 0
                else:
                    self._up_votes[side] = 0
            else:  # up
                if is_up is False:   # 落地
                    self._down_votes[side] += 1
                    if self._down_votes[side] >= self.CONFIRM_DOWN:
                        self.leg_state[side] = "down"
                        self._down_votes[side] = 0
                elif is_up is None:
                    # 膝盖不可见 = 无法判断，清票（不算落地，防拆成2次）
                    self._down_votes[side] = 0
                else:
                    self._down_votes[side] = 0

        # 「抬得不够高」提示: 双腿都判断为"没抬到位"（False 而非 None）时累计
        both_ground_false = (l_lifted is False and r_lifted is False)
        if both_ground_false and 0.55 * torso < d < 0.90 * torso:
            # 膝抬起来了但没到标准高度 -> 提示（持续 ~0.5s 才报）
            self._low_frames += 1
            if self._low_frames == self.LOW_FRAMES:
                event = "hk_height"
        elif not both_ground_false:
            self._low_frames = 0

        any_up = self.leg_state["L"] == "up" or self.leg_state["R"] == "up"
        stage = (T("hk_up", r=d / torso) if any_up
                 else T("hk_down", r=d / torso))
        return Feedback(event=event, count=self.count, stage=stage)


class LungeAnalyzer:
    """弓步蹲（正面视角）：弯曲腿膝角 + 双膝角差 状态机。

    站立 -> 弓步（任一膝角 <140° 或 双膝角差 >25°，持续 2 帧）
         -> 回到站立 -> 计 1 次；弯曲腿最低膝角须 <135°，否则提示蹲得不够低。
    踝不可见时降级为髋角（肩-髋-膝）判定，阈值放宽。"""

    KNEE_STAND = 150
    KNEE_DOWN = 140
    KNEE_VALID = 135
    DIFF_DOWN = 25.0
    HIP_STAND = 150
    HIP_DOWN = 125
    HIP_VALID = 120
    CONFIRM = 2

    def __init__(self):
        self.count = 0
        self.state = "stand"
        self.min_knee = 180.0
        self._votes = 0
        self._reject_note = ""
        self._reject_until = 0.0

    def update(self, k: np.ndarray) -> Feedback:
        if k is None or any(k[KP[n]][2] < KPT_CONF_THRES
                            for n in ("left_hip", "right_hip")):
            return Feedback(stage=T("no_person"), count=self.count)

        lh, rh = k[KP["left_hip"]], k[KP["right_hip"]]
        lk, rk = k[KP["left_knee"]], k[KP["right_knee"]]
        la, ra = k[KP["left_ankle"]], k[KP["right_ankle"]]
        ls, rs = k[KP["left_shoulder"]], k[KP["right_shoulder"]]

        hip_mid = _mid(lh[:2], rh[:2])
        shoulder_mid = _mid(ls[:2], rs[:2])
        torso = float(np.linalg.norm(shoulder_mid - hip_mid)) if (
            ls[2] >= KPT_CONF_THRES and rs[2] >= KPT_CONF_THRES) else 200.0
        y_lo = shoulder_mid[1] if torso < 200 else hip_mid[1] - torso
        y_hi = hip_mid[1] + 1.8 * torso

        # 膝盖用测量阈值 + 范围校验
        knee_ok = (_measure_ok(lk, y_lo, y_hi) and _measure_ok(rk, y_lo, y_hi))
        if not knee_ok:
            if (k is not None and k[KP["left_hip"]][2] >= KPT_CONF_THRES
                    and k[KP["right_hip"]][2] >= KPT_CONF_THRES):
                return Feedback(stage=T("knees_out"), count=self.count)
            return Feedback(stage=T("no_person"), count=self.count)

        ankles_ok = (_measure_ok(la, y_lo, hip_mid[1] + 2.5 * torso) and
                     _measure_ok(ra, y_lo, hip_mid[1] + 2.5 * torso))
        if ankles_ok:
            aL = _angle(lh[:2], lk[:2], la[:2])
            aR = _angle(rh[:2], rk[:2], ra[:2])
            t_stand, t_down, t_valid, t_diff = (self.KNEE_STAND, self.KNEE_DOWN,
                                                self.KNEE_VALID, self.DIFF_DOWN)
            ang_name = T("knee_angle")
        else:
            if ls[2] < KPT_CONF_THRES or rs[2] < KPT_CONF_THRES:
                return Feedback(stage=T("no_person"), count=self.count)
            aL = _angle(ls[:2], lh[:2], lk[:2])
            aR = _angle(rs[:2], rh[:2], rk[:2])
            t_stand, t_down, t_valid, t_diff = (self.HIP_STAND, self.HIP_DOWN,
                                                self.HIP_VALID, 10.0)
            ang_name = T("hip_angle")

        knee = min(aL, aR)
        diff = abs(aL - aR)
        side = T("side_l") if aL <= aR else T("side_r")
        down_sig = knee < t_down or (ankles_ok and diff > t_diff)
        up_sig = aL > t_stand and aR > t_stand and diff < 12

        # 弯腰门闸: 弯腰时双膝角差也会变大，但躯干前倾 >>55°
        if ankles_ok:
            trunk = _angle(_mid(lh[:2], rh[:2]), _mid(ls[:2], rs[:2]),
                           _mid(ls[:2], rs[:2]) + np.array([0.0, 1.0]))
            if trunk >= 55:
                down_sig = False

        event = None
        if self.state == "stand":
            self._votes = self._votes + 1 if down_sig else 0
            if self._votes >= self.CONFIRM:
                self.state = "down"
                self.min_knee = knee
                self._votes = 0
        else:  # down
            self.min_knee = min(self.min_knee, knee)
            self._votes = self._votes + 1 if up_sig else 0
            if self._votes >= self.CONFIRM:
                if self.min_knee < t_valid:
                    self.count += 1
                    event = "count"
                else:
                    event = "lunge_depth"
                    self._reject_note = T("lunge_reject", deg=self.min_knee)
                    self._reject_until = _time.monotonic() + 2.0
                self.state = "stand"
                self._votes = 0

        if self.state == "down":
            stage = T("lunge_down", side=side, a=ang_name, deg=knee)
        elif _time.monotonic() < self._reject_until:
            stage = self._reject_note
        else:
            stage = T("lunge_stand")
        return Feedback(event=event, count=self.count, stage=stage)


class ArmRaiseAnalyzer:
    """侧平举 / 前平举：双腕同时抬至肩线 -> 计 1 次；双臂回落 -> 准备下一次。

    判据（普通人版，体育标准"手臂完全水平"过严——肩关节活动度有限）:
    腕到「肩下 0.15 躯干长」即算举起。
    纠错: 单臂举起不同步 / 双臂停在半程未到位。"""

    RAISE_TOL = 0.15    # 腕在肩下方该躯干长比例以内 = 举起（普通人版）
    LOWER_TOL = 0.40    # 双腕回落到肩下方该比例 = 放下（滞回防抖动）
    HALF_LOW = 0.35     # 半程上限（肩下 0.15~0.35 躯干长之间=抬到一半）
    CONFIRM = 2
    SYNC_FRAMES = 6     # 单臂不同步持续帧数 -> 提示
    HALF_FRAMES = 8     # 半程停留持续帧数 -> 提示

    def __init__(self):
        self.count = 0
        self.state = "down"          # down 双臂下垂 / up 双臂举起
        self._votes = 0
        self._sync_frames = 0
        self._half_frames = 0

    def update(self, k: np.ndarray) -> Feedback:
        need = ("left_shoulder", "right_shoulder",
                "left_hip", "right_hip")
        if k is None or any(k[KP[n]][2] < KPT_CONF_THRES for n in need):
            # 无法判断 = 清确认票，不作为动作证据（防跨丢帧拼凑计次）
            self._votes = 0
            return Feedback(stage=T("no_person"), count=self.count)
        ls, rs = k[KP["left_shoulder"]], k[KP["right_shoulder"]]
        lh, rh = k[KP["left_hip"]], k[KP["right_hip"]]
        shoulder_y = _mid(ls[:2], rs[:2])[1]
        torso = max(_mid(lh[:2], rh[:2])[1] - shoulder_y, 30.0)

        lw, rw = k[KP["left_wrist"]], k[KP["right_wrist"]]
        # 腕用测量阈值 + 范围校验（腕可在肩上方到髋下方之间）
        w_y_lo = shoulder_y - 1.5 * torso
        w_y_hi = shoulder_y + 1.0 * torso
        if not _measure_ok(lw, w_y_lo, w_y_hi) or not _measure_ok(rw, w_y_lo, w_y_hi):
            # 腕不可见 = 无法判断，清票（防 1帧举起→丢人→1帧举起 凑满计次）
            self._votes = 0
            return Feedback(stage=T("wrists_out"), count=self.count)
        upL = lw[1] < shoulder_y + self.RAISE_TOL * torso
        upR = rw[1] < shoulder_y + self.RAISE_TOL * torso
        lowL = lw[1] > shoulder_y + self.LOWER_TOL * torso
        lowR = rw[1] > shoulder_y + self.LOWER_TOL * torso

        event = None
        if self.state == "down":
            both_up = upL and upR
            self._votes = self._votes + 1 if both_up else 0
            if self._votes >= self.CONFIRM:
                self.state = "up"
                self.count += 1
                event = "count"
                self._votes = 0
                self._sync_frames = 0
                self._half_frames = 0
            elif not both_up:
                # 纠错: 单臂举起（另一侧没跟上）
                if upL != upR:
                    self._sync_frames += 1
                    self._half_frames = 0
                    if self._sync_frames == self.SYNC_FRAMES:
                        event = "raise_sync"
                else:
                    self._sync_frames = 0
                    # 纠错: 双臂抬到一半停住
                    d = min(lw[1], rw[1]) - shoulder_y
                    if self.RAISE_TOL * torso < d < self.HALF_LOW * torso:
                        self._half_frames += 1
                        if self._half_frames == self.HALF_FRAMES:
                            event = "raise_height"
                    else:
                        self._half_frames = 0
        else:  # up
            both_low = lowL and lowR
            self._votes = self._votes + 1 if both_low else 0
            if self._votes >= self.CONFIRM:
                self.state = "down"
                self._votes = 0

        stage = T("raise_up") if self.state == "up" else T("raise_down")
        return Feedback(event=event, count=self.count, stage=stage)


class HorseStanceAnalyzer:
    """马步蹲（计时模式）：膝角保持低位 -> 累计坚持秒数，中断不计损失。

    阈值（普通人版，武术标准"大腿平行地面 90~100°"过严，蹲太低坚持不了
    几秒没有成就感）: 膝角 <135° 即视为保持中。
    每 10/20/30/45/60 秒触发里程碑事件（UI 播报）；膝角不够低持续 ->
    语音提醒蹲低。踝不可见时降级为髋角判定。"""

    KNEE_HOLD = 135     # 膝角低于此 = 马步保持中（普通人版）
    KNEE_WARN = 150     # (HOLD, WARN) 之间 = 想蹲但不够低
    FOOT_WIDE = 1.1     # 踝距 > 1.1 肩宽 = 双脚分开（并腿蹲不算马步）
    HIP_HOLD = 125
    HIP_WARN = 140
    MILESTONES = (10, 20, 30, 45, 60)
    WARN_FRAMES = 10
    # 高度兜底模式（膝盖出镜时）: 髋下沉超过该躯干长比例 = 保持中
    # （膝角135° ≈ 下沉 0.24 躯干长，实际蹲马步多在 120~135°，取 0.28 兼顾）
    H_HOLD_ENTER = 0.28
    H_HOLD_EXIT = 0.20

    def __init__(self):
        self.seconds = 0.0           # 已累计坚持秒数
        self._t_start = None         # 本段保持开始时刻
        self._warn_frames = 0
        self._announced = set()
        # 高度兜底模式（膝盖出镜时用）
        self._hip_base = None        # 站立时髋部 y 基线（EMA）
        self._torso_ref = None       # 站立时躯干长参考（EMA）
        # 连续上半身可见帧数：高度兜底要求连续可见才判定，防单帧闪现误计
        self._upper_visible = 0

    def _display(self) -> float:
        extra = (_time.monotonic() - self._t_start) if self._t_start else 0.0
        return self.seconds + extra

    def _milestone(self, disp: int):
        """跨越里程碑阈值时返回 (事件, 播报值)；否则 (None, disp)"""
        for m in self.MILESTONES:
            if disp >= m and m not in self._announced:
                self._announced.add(m)
                return "count", m
        return None, disp

    def _calib_base(self, hip_y: float, torso: float, window: float = 0.3):
        """站立时校准髋基线 + 躯干长（EMA）—— 高度兜底用，防基线追赶。

        window: 校准窗口（躯干长比例）；马步蹲得浅（下沉约 0.28），
        须用比下沉量更窄的窗口（0.15），否则蹲着时基线会往下漂。"""
        if self._hip_base is None:
            self._hip_base, self._torso_ref = hip_y, torso
            return
        d = hip_y - self._hip_base
        if d < -0.5 * self._torso_ref:
            # 基线偏低（启动时人在低位）-> 直接重锚
            self._hip_base, self._torso_ref = hip_y, torso
        elif abs(d) < window * self._torso_ref:
            # 仅在"确实站着"时校准 —— 防止下蹲过程中基线追着髋部跑
            self._hip_base += 0.15 * (hip_y - self._hip_base)
            self._torso_ref += 0.15 * (torso - self._torso_ref)

    def _hold_by_height(self, k: np.ndarray) -> Feedback:
        """高度兜底: 膝盖出镜（踝必然也出镜）时用髋部下沉量判定保持中。

        肩/髋可见即可工作；宽度无法校验（信任用户在做马步）。"""
        lh, rh = k[KP["left_hip"]], k[KP["right_hip"]]
        ls, rs = k[KP["left_shoulder"]], k[KP["right_shoulder"]]
        hip_y = _mid(lh[:2], rh[:2])[1]
        torso = max(hip_y - _mid(ls[:2], rs[:2])[1], 30.0)
        now = _time.monotonic()

        self._calib_base(hip_y, torso, window=0.15)

        drop = hip_y - self._hip_base
        holding = drop > (self.H_HOLD_ENTER if self._t_start is None
                          else self.H_HOLD_EXIT) * self._torso_ref
        if holding:
            if self._t_start is None:
                self._t_start = now
            self._warn_frames = 0
        else:
            if self._t_start is not None:
                self.seconds += now - self._t_start
                self._t_start = None

        disp = int(self._display())
        event, disp = self._milestone(disp)
        if self._t_start is not None:
            stage = T("horse_hold_h", n=disp)
        elif self.seconds > 0.5:
            stage = T("horse_rest", n=int(self.seconds))
        else:
            stage = T("knees_out")
        return Feedback(event=event, count=disp, stage=stage)

    def update(self, k: np.ndarray) -> Feedback:
        if k is None or any(k[KP[n]][2] < KPT_CONF_THRES
                            for n in ("left_hip", "right_hip")):
            # 无人/上半身缺失 -> 视为中断，结算已坚持秒数
            self._upper_visible = 0
            if self._t_start is not None:
                self.seconds += _time.monotonic() - self._t_start
                self._t_start = None
            return Feedback(stage=T("no_person"), count=int(self.seconds))
        self._upper_visible += 1

        lh, rh = k[KP["left_hip"]], k[KP["right_hip"]]
        lk, rk = k[KP["left_knee"]], k[KP["right_knee"]]
        la, ra = k[KP["left_ankle"]], k[KP["right_ankle"]]
        ls, rs = k[KP["left_shoulder"]], k[KP["right_shoulder"]]

        hip_mid = _mid(lh[:2], rh[:2])
        shoulder_mid = _mid(ls[:2], rs[:2]) if (
            ls[2] >= KPT_CONF_THRES and rs[2] >= KPT_CONF_THRES) else None
        torso = max(hip_mid[1] - shoulder_mid[1], 30.0) if shoulder_mid is not None else 200.0

        # 持续校准髋基线
        if shoulder_mid is not None:
            self._calib_base(hip_mid[1], torso, window=0.15)

        # 膝盖用测量阈值 + 范围校验
        y_lo = (shoulder_mid[1] if shoulder_mid is not None else hip_mid[1] - torso)
        y_hi = hip_mid[1] + 1.8 * torso
        knee_ok = (_measure_ok(lk, y_lo, y_hi) and _measure_ok(rk, y_lo, y_hi))
        if not knee_ok:
            # 膝出镜但肩髋仍可见 -> 高度兜底计时
            # 要求连续 2 帧上半身可见才进入兜底，防单帧闪现+基线漂移误计
            if shoulder_mid is not None and self._upper_visible >= 2:
                return self._hold_by_height(k)
            if self._t_start is not None:
                self.seconds += _time.monotonic() - self._t_start
                self._t_start = None
            return Feedback(stage=T("no_person"), count=int(self.seconds))

        ankles_ok = (_measure_ok(la, y_lo, hip_mid[1] + 2.5 * torso) and
                     _measure_ok(ra, y_lo, hip_mid[1] + 2.5 * torso))
        if ankles_ok:
            ang = (_angle(lh[:2], lk[:2], la[:2]) +
                   _angle(rh[:2], rk[:2], ra[:2])) / 2
            t_hold, t_warn, ang_name = self.KNEE_HOLD, self.KNEE_WARN, T("knee_angle")
            # 马步需双脚分开（踝可见时校验；并腿微蹲不计入保持）
            foot_w = abs(ra[0] - la[0])
            shoulder_w = abs(rs[0] - ls[0]) + 30
            if foot_w < shoulder_w * self.FOOT_WIDE:
                if self._t_start is not None:
                    self.seconds += _time.monotonic() - self._t_start
                    self._t_start = None
                return Feedback(event=None, count=int(self._display()),
                                 stage=T("horse_wide"))
        else:
            if ls[2] < KPT_CONF_THRES or rs[2] < KPT_CONF_THRES:
                if self._t_start is not None:
                    self.seconds += _time.monotonic() - self._t_start
                    self._t_start = None
                return Feedback(stage=T("no_person"), count=int(self.seconds))
            ang = (_angle(ls[:2], lh[:2], lk[:2]) +
                   _angle(rs[:2], rh[:2], rk[:2])) / 2
            t_hold, t_warn, ang_name = self.HIP_HOLD, self.HIP_WARN, T("hip_angle")

        event = None
        if ang < t_hold:
            if self._t_start is None:
                self._t_start = _time.monotonic()
            self._warn_frames = 0
        else:
            if self._t_start is not None:      # 结束本段保持
                self.seconds += _time.monotonic() - self._t_start
                self._t_start = None
            if ang < t_warn:                   # 想蹲但不够低 -> 提醒
                self._warn_frames += 1
                if self._warn_frames == self.WARN_FRAMES:
                    event = "horse_depth"
            else:
                self._warn_frames = 0

        disp = int(self._display())
        ms_event, disp = self._milestone(disp)
        if ms_event:
            event = ms_event

        if self._t_start is not None:
            stage = T("horse_hold", n=disp, a=ang_name, deg=ang)
        elif self.seconds > 0.5:
            stage = T("horse_rest", n=int(self.seconds))
        else:
            stage = T("horse_ready")
        return Feedback(event=event, count=disp, stage=stage)


# 动作注册表: key -> {"cls": 分析器, "sound": 切换语音名}；顺序即下拉框顺序
EXERCISES = {
    "squat":        {"cls": SquatAnalyzer,       "sound": "select_squat"},
    "jumping_jack": {"cls": JumpingJackAnalyzer, "sound": "select_jumping_jack"},
    "high_knees":   {"cls": HighKneesAnalyzer,   "sound": "select_high_knees"},
    "lunge":        {"cls": LungeAnalyzer,       "sound": "select_lunge"},
    "arm_raise":    {"cls": ArmRaiseAnalyzer,    "sound": "select_arm_raise"},
    "horse_stance": {"cls": HorseStanceAnalyzer, "sound": "select_horse_stance"},
}

# 动作名（UI 下拉框显示）
EXERCISE_NAMES = {
    "squat":        {"zh": "深蹲",   "en": "Squat"},
    "jumping_jack": {"zh": "开合跳", "en": "Jumping Jacks"},
    "high_knees":   {"zh": "高抬腿", "en": "High Knees"},
    "lunge":        {"zh": "弓步蹲", "en": "Lunge"},
    "arm_raise":    {"zh": "侧平举", "en": "Arm Raise"},
    "horse_stance": {"zh": "马步蹲", "en": "Horse Stance"},
}
