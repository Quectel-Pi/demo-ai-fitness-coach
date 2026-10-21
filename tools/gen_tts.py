#!/usr/bin/env python3
"""用 edge-tts（微软神经语音）在主机上合成中/英文语音，板上离线播放 wav。

选型说明（2026-10 语音重做）:
  - 板上运行完全离线（只播 wav），TTS 只在主机生成时联网 -> 用 edge-tts 拿到
    远超离线方案（piper 等）的音质与情感表现（社区评测一致结论）
  - 中文音色: zh-CN-XiaoxiaoNeural（晓晓，温暖亲切女声）
    英文音色: en-US-JennyNeural（Jenny，热情自然女声，社区评测英文最佳之一）
  - 热情感: 语速 +15%、音调 +8Hz（试听调优结论）
  - 依赖（可装进临时目录，不污染宿主机）:
      pip install --target ./edge-tts-py edge-tts miniaudio
      PYTHONPATH=./edge-tts-py python3 tools/gen_tts.py [zh|en]

用法: PYTHONPATH=./edge-tts-py python3 tools/gen_tts.py [zh|en]
      不带参数 = 两种语言全部生成
输出: ../audio/<lang>/<name>.wav (22050 等不合板端，paplay 自动重采样)
"""
import subprocess
import sys
import wave
from pathlib import Path

TMP = Path("/home/q/work/temp/tmp/fitness-coach")
AUDIO_ROOT = Path(__file__).resolve().parent.parent / "audio"

# 每种语言: (edge-tts 音色, 话术字典)。话术键名与 audio_player 播放名一致。
LANGS = {
    "zh": {
        "voice": "zh-CN-XiaoxiaoNeural",
        "rate": "+15%",
        "pitch": "+8Hz",
        "phrases": {
            "welcome": "欢迎来到智能健身教练！我在看着你哦，快选一个动作开始吧！",
            "select_squat": "深蹲模式",
            "select_jumping_jack": "开合跳模式",
            "select_high_knees": "高抬腿模式",
            "select_lunge": "弓步蹲模式",
            "select_arm_raise": "侧平举模式",
            "select_horse_stance": "马步蹲模式",
            "start": "三、二、一，开始",
            "stop": "好的，先休息一下！",
            "finish": "哇！太厉害了！今天的状态简直满分！",
            "no_person": "咦？好像没看到你，站到摄像头前面就能开始啦！",   # 备用（当前逻辑不主动催促）
            "person_back": "我来啦！已经看到你了，随时可以开始！",
            # 深蹲纠错
            "squat_knee": "注意哦！膝盖不要内扣！",
            "squat_depth": "再蹲深一点！你可以的！",
            "squat_back": "背部挺起来！你是最棒的！",
            # 开合跳纠错
            "jj_arms": "手臂甩起来！举过头顶！",
            "jj_feet": "双脚打开！再大一点！",
            # 高抬腿纠错
            "hk_height": "膝盖再抬高一点！",
            # 弓步蹲纠错
            "lunge_depth": "再蹲低一点！后腿弯下去！",
            # 侧平举纠错
            "raise_sync": "双臂一起抬！",
            "raise_height": "手臂抬平！举到肩膀高度！",
            # 马步蹲纠错 + 计时里程碑
            "horse_depth": "再蹲低一点！坚持住！",
            "hs_m10": "已经坚持十秒了！真棒！",
            "hs_m20": "二十秒！稳住，你是最强的！",
            "hs_m30": "三十秒达成！太厉害了！",
            "hs_m45": "四十五秒！简直是钢铁大腿！",
            "hs_m60": "一分钟！满分坚持！",
            # 鼓励（随机轮换）
            "good_1": "漂亮！",
            "good_2": "完美！继续！",
            "good_3": "太标准了！",
            "good_4": "就是这样！",
            "milestone_10": "太棒了！已经十个了！你的坚持我看在眼里！",
            "milestone_20": "二十个达成！简直是健身达人！",
            "rest": "休息一下，喝口水，待会儿继续战斗！",
        },
    },
    "en": {
        "voice": "en-US-JennyNeural",
        "rate": "+10%",
        "pitch": "+4Hz",
        "phrases": {
            "welcome": "Welcome to your AI fitness coach! I'm watching — pick an exercise and let's go!",
            "select_squat": "Squat mode",
            "select_jumping_jack": "Jumping jacks mode",
            "select_high_knees": "High knees mode",
            "select_lunge": "Lunge mode",
            "select_arm_raise": "Arm raise mode",
            "select_horse_stance": "Horse stance mode",
            "start": "Three, two, one, go!",
            "stop": "Okay, take a break!",
            "finish": "Wow! Amazing! You crushed it today!",
            "no_person": "Hmm, I can't see you — step in front of the camera to start!",
            "person_back": "There you are! Ready when you are!",
            "squat_knee": "Watch it! Don't let your knees cave in!",
            "squat_depth": "Squat deeper! You've got this!",
            "squat_back": "Keep your back straight! You're doing great!",
            "jj_arms": "Swing those arms! Over your head!",
            "jj_feet": "Feet wider! Come on!",
            "hk_height": "Lift those knees higher!",
            "lunge_depth": "Go lower! Bend that back leg!",
            "raise_sync": "Both arms together!",
            "raise_height": "Arms up! To shoulder height!",
            "horse_depth": "Lower! Hold it!",
            "hs_m10": "Ten seconds! Nice!",
            "hs_m20": "Twenty seconds! Stay strong!",
            "hs_m30": "Thirty seconds! Incredible!",
            "hs_m45": "Forty-five seconds! Steel legs!",
            "hs_m60": "One minute! Perfect hold!",
            "good_1": "Nice!",
            "good_2": "Perfect! Keep going!",
            "good_3": "Textbook form!",
            "good_4": "That's it!",
            "milestone_10": "Ten reps! I see your effort!",
            "milestone_20": "Twenty reps! You're a machine!",
            "rest": "Take a rest, grab some water, we'll get back at it!",
        },
    },
}


def synth(lang: str, name: str, text: str):
    """edge-tts 出 mp3 -> miniaudio 解码降混为单声道 wav"""
    cfg = LANGS[lang]
    out_dir = AUDIO_ROOT / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    mp3 = TMP / f"_tts_{lang}_{name}.mp3"
    wav = out_dir / f"{name}.wav"
    subprocess.run(
        [sys.executable, "-m", "edge_tts",
         "--voice", cfg["voice"], "--rate", cfg["rate"], "--pitch", cfg["pitch"],
         "--text", text, "--write-media", str(mp3)],
        check=True, capture_output=True,
        env={"PYTHONPATH": str(TMP / "edge-tts-py"), "PATH": "/usr/bin:/bin"},
    )
    import miniaudio
    dec = miniaudio.decode_file(str(mp3))
    samples = dec.samples
    if dec.nchannels == 2:
        samples = samples[0::2]
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(dec.sample_rate)
        w.writeframes(samples)
    mp3.unlink()
    print(f"[ok:{lang}] {name}: {text}")


def main():
    langs = sys.argv[1:] or ["zh", "en"]
    for lang in langs:
        cfg = LANGS[lang]
        tasks = dict(cfg["phrases"])
        for i in range(1, 31):  # 计数 1~30
            tasks[f"num_{i}"] = str(i)
        out_dir = AUDIO_ROOT / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, text in tasks.items():
            if (out_dir / f"{name}.wav").exists():
                continue
            synth(lang, name, text)


if __name__ == "__main__":
    main()
