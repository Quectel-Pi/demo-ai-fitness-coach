#!/usr/bin/env python3
"""语音播报：队列串行播放 audio/<lang>/ 下的 wav，纠错提示带冷却。

音频按语言分目录: audio/zh/（晓晓）、audio/en/（Jenny），set_lang 切换。
板上播放链路: paplay 走系统音频路由（跟随用户选择的输出设备，含蓝牙耳机）
             -> 失败时回退 aplay 直出 ALSA es8389（root 会话下 PipeWire 拒连）。
在独立线程中运行，避免阻塞 UI / 推理。
"""
import queue
import subprocess
import threading
import time
from pathlib import Path

AUDIO_ROOT = Path(__file__).resolve().parent.parent / "audio"

# 各类提示的冷却时间（秒）：冷却期内同一提示不重复播报
COOLDOWN = 6.0


class AudioPlayer:
    def __init__(self, lang: str = "zh", event_log: list = None):
        self.lang = lang
        # 播报事件日志: 实际开始播放时 append {"t": monotonic, "wav": path}
        # 供录屏后混音对齐（demo 录制用；None 则不记录）
        self.event_log = event_log
        self.q: "queue.Queue[str]" = queue.Queue()
        self._last_play: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._proc = None
        self._killed = False
        self._thread.start()

    def set_lang(self, lang: str):
        """切换语言目录（切换瞬间清空冷却记录，避免新语言提示被误抑制）"""
        if lang != self.lang:
            self.lang = lang
            self._last_play.clear()

    def play(self, name: str, cooldown: bool = False):
        """入队播报。cooldown=True 时同类提示在冷却期内被丢弃。"""
        wav = AUDIO_ROOT / self.lang / f"{name}.wav"
        if not wav.exists():
            return
        now = time.monotonic()
        if cooldown:
            if now - self._last_play.get(name, -1e9) < COOLDOWN:
                return
            self._last_play[name] = now
        self.q.put(str(wav))

    def stop_current(self):
        """打断当前正在播的（切动作/结束时）"""
        if self._proc and self._proc.poll() is None:
            self._killed = True
            self._proc.kill()

    def clear_queue(self):
        """丢弃所有待播语音并打断当前正在播的（切换动作时用）"""
        with self.q.mutex:
            self.q.queue.clear()
        self.stop_current()

    def shutdown(self):
        self._stop.set()
        self.stop_current()
        self._thread.join(timeout=1.0)   # 确保事件日志写完（meta.json 落盘前）

    def _try(self, cmd) -> bool:
        """播一条；返回是否成功（被主动打断视为已处理）"""
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._proc.wait(timeout=10)
        except Exception:
            return False
        if self._killed:          # stop_current 打断的，不重播
            self._killed = False
            return True
        return self._proc.returncode == 0

    def _run(self):
        while not self._stop.is_set():
            try:
                wav = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            # 记录实际播放时刻（出队点 = 播放开始点，误差为进程启动开销）
            if self.event_log is not None:
                self.event_log.append({"t": time.monotonic(), "wav": wav})
            # 1) 系统路由: 声音跟随系统设置的输出设备（蓝牙耳机连接后自动进耳机）
            if self._try(["paplay", wav]):
                continue
            # 2) 兜底: root/无桌面会话时 PipeWire 拒连 -> 直出硬件声卡
            self._try(["aplay", "-q", "-D", "plughw:0,0", wav])
