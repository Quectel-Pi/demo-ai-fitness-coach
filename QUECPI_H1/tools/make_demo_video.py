#!/usr/bin/env python3
"""录屏素材合成（宿主机工具）: demo_out/(帧序列 + meta.json) -> 演示视频 MP4

用法:
  python3 tools/make_demo_video.py --dir demo_out --out demo.mp4 \
      [--narration narration.json] [--srt subs.srt] [--ffmpeg /path/to/ffmpeg]

  --dir        板上录屏产物目录（frame_00000.jpg... + meta.json）
  --out        输出 MP4 路径
  --narration  旁白脚本 JSON: [{"t": 秒, "text": "..."}]，
               用 edge-tts 逐条合成后按时间轴混入（edge-tts 不可用时跳过并警告）
  --srt        SRT 字幕文件（烧录进画面，需 ffmpeg 带 libass）
  --ffmpeg     ffmpeg 可执行文件路径（默认取 PATH 或 imageio_ffmpeg）

合成流程:
  A) 帧序列 -> H264（此步顺带烧字幕）
  B) 播报事件(meta.audio_events) + 旁白 按各自时间点 adelay 混音 -> AAC
     视频流 copy 不重编码，音频与画面按录制时间轴对齐
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_ffmpeg(explicit: str) -> str:
    if explicit:
        return explicit
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    sys.exit("[compose] 找不到 ffmpeg（PATH/ imageio_ffmpeg 均无），用 --ffmpeg 指定")


def probe_duration(ffmpeg: str, path: str) -> float:
    """ffprobe 不一定有，用 ffmpeg -i 的 stderr 解析时长"""
    r = subprocess.run([ffmpeg, "-i", path], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "Duration:" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def tts_narration(items: list, workdir: Path) -> list:
    """edge-tts 逐条合成旁白，返回 [(t, mp3_path)]；不可用时返回 []"""
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        print("[compose] WARN: 未安装 edge-tts，跳过旁白（pip install edge-tts）")
        return []
    import asyncio
    import edge_tts

    out = []
    for i, it in enumerate(items):
        mp3 = workdir / f"nar_{i:02d}.mp3"
        asyncio.run(edge_tts.Communicate(
            it["text"], "zh-CN-XiaoxiaoNeural").save(str(mp3)))
        out.append((float(it["t"]), str(mp3)))
        print(f"[tts] {it['t']:6.1f}s {it['text'][:24]}...")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="板上录屏产物目录")
    ap.add_argument("--out", required=True, help="输出 MP4")
    ap.add_argument("--narration", help="旁白 JSON [{t, text}]")
    ap.add_argument("--srt", help="SRT 字幕（烧录）")
    ap.add_argument("--ffmpeg", help="ffmpeg 路径")
    ap.add_argument("--voice", default="zh-CN-XiaoxiaoNeural",
                    help="edge-tts 音色（默认晓晓）")
    args = ap.parse_args()

    ffmpeg = find_ffmpeg(args.ffmpeg)
    d = Path(args.dir)
    meta = json.loads((d / "meta.json").read_text())
    fps = meta["record_fps"]
    frames = sorted(d.glob("frame_*.jpg"))
    if not frames:
        sys.exit(f"[compose] {d} 下没有帧序列")
    dur = len(frames) / fps
    print(f"[compose] {len(frames)} frames @ {fps}fps = {dur:.1f}s")

    work = Path(tempfile.mkdtemp(prefix="demo-compose-", dir="/home/q/work/temp/tmp"))
    silent = work / "silent.mp4"

    # ---- A: 帧序列 -> H264（顺带烧字幕） ----
    # 偶数化: 窗口缩放后可能出奇数高（如 1280x809），yuv420p 编码要求宽高为偶数
    vf_chain = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    if args.srt:
        srt = str(Path(args.srt).resolve()).replace("\\", "/").replace(":", "\\:")
        fonts = "/usr/share/fonts"
        vf_chain += (f",subtitles='{srt}':fontsdir='{fonts}':"
                     f"force_style='FontName=Noto Sans CJK SC,FontSize=15,"
                     f"PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=1'")
    vf = ["-vf", vf_chain]
    cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", str(d / "frame_%05d.jpg"),
           "-t", f"{dur:.3f}", "-c:v", "libx264", "-crf", "19",
           "-pix_fmt", "yuv420p", *vf, str(silent)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[compose] 帧合成失败:\n{r.stderr[-2000:]}")
    print(f"[compose] silent video ok")

    # ---- B: 混音 ----
    # 音频源: 播报事件 wav + 旁白 mp3，各自 adelay 到录制时间轴；基底静音轨定长
    audio_inputs = []          # [(t, path)]
    for ev in meta.get("audio_events", []):
        p = Path(ev["wav"])
        # 板上路径 -> 宿主机: wav 在安装包 audio/ 下；目录对不上时按名字在
        # 本仓库 AI-Fitness-Coach/audio/ 下找回（事件日志记录的是板上绝对路径）
        if not p.exists():
            cand = Path(__file__).resolve().parent.parent / "audio" / \
                p.parent.name / p.name
            if cand.exists():
                p = cand
            else:
                print(f"[compose] WARN: 缺音频 {p.name}，跳过")
                continue
        audio_inputs.append((float(ev["t"]), str(p)))
    if args.narration:
        audio_inputs += tts_narration(
            json.loads(Path(args.narration).read_text()), work)

    if not audio_inputs:
        shutil.copy(silent, args.out)
        print(f"[compose] 无音频 -> {args.out}")
        return

    base = len(audio_inputs)          # 0 号输入是 silent.mp4，1 号静音基底，2 号起音频
    # 基底: 静音轨保证 amix 输出时长=视频时长
    fc_parts = [f"[1:a]aformat=channel_layouts=stereo,apad=whole_dur={dur:.3f}[base]"]
    mix_labels = ["[base]"]
    for i, (t, _) in enumerate(audio_inputs):
        ms = max(0, int(round(t * 1000)))
        fc_parts.append(
            f"[{2 + i}:a]aformat=channel_layouts=stereo,"
            f"adelay={ms}|{ms}[a{i}]")
        mix_labels.append(f"[a{i}]")
    fc_parts.append(
        f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:"
        f"duration=first:normalize=0[aout]")

    cmd = [ffmpeg, "-y", "-i", str(silent),
           "-f", "lavfi", "-t", f"{dur:.3f}",
           "-i", "anullsrc=r=44100:cl=stereo"]
    for _, p in audio_inputs:
        cmd += ["-i", p]
    cmd += ["-filter_complex", ";".join(fc_parts),
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest",
            str(Path(args.out).resolve())]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[compose] 混音失败:\n{r.stderr[-2000:]}")
    print(f"[compose] done -> {args.out} "
          f"({len(audio_inputs)} audio tracks mixed)")


if __name__ == "__main__":
    main()
