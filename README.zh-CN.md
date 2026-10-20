# 智能健身教练

[English](README.md) | 简体中文

智能健身教练是基于 Quectel Pi（RK3576）的边缘 AI 健身应用。通过 USB 摄像头采集画面，在板载 NPU 上运行 YOLO11n-pose 实时追踪 17 个人体关键点，自动完成六种动作的计数与计时，并以中英双语语音进行教练播报——全程离线运行。

![应用预览](assets/app-preview.jpg)

## 功能特性

- **实时姿态识别** — YOLO11n-pose（RKNN，fp16）跑在 RK3576 NPU 上，单帧推理约 55 ms
- **六种动作** — 深蹲 / 开合跳 / 高抬腿 / 弓步蹲 / 侧平举（计次）+ 马步蹲（计时）
- **防误计门闸** — 多帧确认 + 深度 / 躯干前倾阈值，有效剔除半程动作与误计数
- **动作纠错提示** — 动作未计次时，原因在界面上停留 2 秒（如「上次蹲不够深(121°) 未计次」）
- **内置示范视频** — 每种动作附带标准示范视频，应用内播放器播完自动收起
- **数字变焦** — 0.5×–3×、步进 0.25×（界面按钮或 `+` / `-` / `0` 快捷键），识别在变焦后的画面上进行
- **中英双语** — 界面与语音一键切换，离线语音包随包内置
- **训练控制** — 待机门闸（点「开始训练」前不计数）、暂停冻结计数、一键归零

## 技术栈

- Python 3 + PySide6（Qt）界面，QPainter 绘制骨架叠加
- YOLO11n-pose 转换为 RKNN（fp16 不量化），经 librknnrt 在 NPU 上推理
- OpenCV V4L2 采集，预览镜像翻转
- edge-tts 生成的离线语音包，PipeWire 播放、ALSA 兜底
- Go 静态启动器 + `.run` 自解压安装包

## 硬件要求

| 项目 | 要求 |
|---|---|
| 板子 | Quectel Pi（RK3576，6 TOPS NPU） |
| 摄像头 | USB UVC 摄像头，建议 720p 及以上 |
| 音频 | 扬声器或耳机（跟随系统输出路由，支持蓝牙耳机） |
| 存储 | 安装约需 1 GB 空间 |

## 项目结构

```
├── code/                    # 应用源码
│   ├── main.py              # 界面、摄像头循环、训练控制
│   ├── exercise_analyzer.py # 六种动作分析器与计次规则
│   ├── pose_estimator.py    # 关键点后处理
│   ├── rknn_backend.py      # RKNN NPU 推理封装
│   ├── audio_player.py      # 语音播放
│   ├── run.sh               # 启动脚本（Wayland 会话发现）
│   └── ai-fitness-coach     # 双击启动器（Go 静态 ELF）
├── models/                  # YOLO11n-pose ONNX + RKNN 模型
├── audio/                   # 离线语音包（中 / 英）
├── assets/refs/             # 动作示范视频
└── tools/                   # 运行时构建与打包脚本
```

## 快速开始

### 1. 获取安装包

从 [Releases](https://github.com/MAODEHUANXIANG/demo-ai-fitness-coach/releases) 页面下载 `ai-fitness-coach-installer.run`（约 220 MB，内置 Python 运行时、模型、语音包与示范视频）。

### 2. 推送到板子

```bash
adb push ai-fitness-coach-installer.run /home/q/
adb shell chown q:q /home/q/ai-fitness-coach-installer.run
```

### 3. 安装

```bash
# 普通用户 q（推荐）→ ~/ai-fitness-coach
adb shell su - q -c '/home/q/ai-fitness-coach-installer.run'

# 或 root → /root/ai-fitness-coach
adb shell /home/q/ai-fitness-coach-installer.run
```

### 4. 启动

在文件管理器中双击 `code/ai-fitness-coach`，或：

```bash
cd ~/ai-fitness-coach/code && ./run.sh
```

## 使用方法

1. 打开动作列表选择动作 — 点任意条目旁的 **▶** 可预览该动作的示范视频
2. 按 **开始训练** — 立即开放计次
3. 面对摄像头训练 — 自动计数并语音播报，未计次的动作会在界面显示原因
4. 底部按钮随时可用：暂停 / 继续、归零、数字变焦、中英切换
5. 训练结束按 **结束训练**

### 计次规则

| 动作 | 计一次的判定 |
|---|---|
| 深蹲 | 膝角低于约 115° 后回到 145° 以上，躯干前倾 < 55° |
| 开合跳 | 双脚分开超过 1.35 倍肩宽且手腕过头顶，再回到并拢 |
| 高抬腿 | 任一膝盖抬至髋部高度，再落地还原 |
| 弓步蹲 | 前弓腿膝角低于约 135°，上身保持挺直 |
| 侧平举 | 双腕同时抬过肩线，再双臂回落 |
| 马步蹲 | 双脚超过 1.1 倍肩宽且膝角低于约 135° 开始计时，中途起身不清零 |

## 常见问题

**检测不到摄像头？**

在板上执行 `ls /dev/video*` 确认设备，重新插拔 USB 摄像头，插入后等待几秒再启动应用。

**动作不计数？**

应用只计标准动作。查看界面上的未计次原因（如蹲得不够深），距摄像头 2.5~3.5 米、全身入镜再试。

**没有语音？**

应用跟随系统音频输出路由。检查系统设置的输出设备，蓝牙耳机同样可用。

## 技术说明

模型转换（ONNX → RKNN）、运行时打包与完整防误计参数，详见 [TECH.md](TECH.md)。
