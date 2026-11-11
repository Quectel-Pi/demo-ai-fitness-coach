# 智能健身教练

智能健身教练是基于 Quectel QuecPi H1（高通 QCS6490）的边缘 AI 健身应用。通过 USB 摄像头采集画面，YOLO11n-pose 双段推理跑在 Hexagon NPU（HTP v68）+ CPU 大核上实时追踪 17 个人体关键点，自动完成六种动作的计数与计时，并以中英双语语音进行教练播报——全程离线运行。

## 功能特性

- **双段 NPU 推理** — 主干 QDQ int8 走 Hexagon NPU（约 6.6 ms）+ 检测头 fp32 走 CPU 大核（约 36 ms），合计约 43 ms/帧，UI 约 13~17 FPS
- **六种动作** — 深蹲 / 开合跳 / 高抬腿 / 弓步蹲 / 侧平举（计次）+ 马步蹲（计时）
- **防误计门闸** — 多帧确认 + 深度 / 躯干前倾阈值 + 踝丢失自动降级（髋角 / 高度兜底），有效剔除半程动作与误计数
- **动作纠错提示** — 动作未计次时，原因在界面上停留 2 秒（如「上次蹲不够深(121°) 未计次」）
- **内置示范视频** — 每种动作附带标准示范视频，应用内播放器播完自动收起（马步蹲为静态定型图视频）
- **数字变焦** — 0.5×–3×、步进 0.25×（界面按钮或 `+` / `-` / `0` 快捷键），识别在变焦后的画面上进行
- **中英双语** — 界面与语音一键切换（晓晓 / Jenny），离线语音包随包内置
- **训练控制** — 待机门闸（点「开始训练」前不计数）、暂停冻结计数、一键归零

## 技术栈

- Python 3 + PySide6（Qt）界面，QPainter 绘制骨架叠加，无框圆角渐变窗口
- YOLO11n-pose 双段推理：主干 QDQ int8 经 QNN Execution Provider 走 libQnnHtp（HTP burst 模式），检测头 fp32 走 CPU（run.sh 自动绑大核 4-7）
- OpenCV V4L2 采集（运行时扫描自动发现摄像头），预览镜像翻转
- edge-tts 生成的离线语音包，pw-play 系统路由优先、ALSA 兜底
- Go 静态启动器 + `.run` 自解压安装包

## 硬件要求

| 项目 | 要求 |
|---|---|
| 板子 | QuecPi H1（高通 QCS6490，Hexagon NPU HTP v68），Debian 13 + GNOME Wayland |
| 摄像头 | USB UVC 摄像头，640x480 即可（YUYV 定焦机型实测可用） |
| 显示 | 板载触摸屏（无框圆角窗口自适应屏幕尺寸） |
| 音频 | 扬声器或耳机（跟随系统输出路由） |
| 存储 | 安装约需 1.5 GB 空间 |

> 权限提示：普通用户需加入 `video`、`audio` 组（`sudo usermod -aG video,audio 用户名` 后重新登录）。

## 项目结构

```
├── code/                    # 应用源码
│   ├── main.py              # 界面、摄像头循环、训练控制
│   ├── exercise_analyzer.py # 六种动作分析器与计次规则
│   ├── pose_estimator.py    # 双后端推理（qnn NPU / onnx CPU 兜底）
│   ├── audio_player.py      # 语音播放
│   ├── run.sh               # 启动脚本（大核绑定/旧实例清理/会话发现）
│   └── ai-fitness-coach     # 双击启动器（Go 静态 ELF）
├── models/                  # YOLO11n-pose 双段模型（QDQ 主干 + fp32 检测头 + CPU 兜底整模型）
├── audio/                   # 离线语音包（中 zh / 英 en，各 64 条）
├── assets/refs/             # 动作示范视频
├── runtime/                 # Qt 运行时（构建产物，仓库不含）
└── tools/                   # 运行时构建与打包脚本
```

## 快速开始

### 1. 获取安装包

从项目 Releases 页面下载 `ai-fitness-coach-installer.run`（约 222 MB，自解压格式，内置 Python 运行时、AI 模型、中英文语音包与示范视频）。

### 2. 推送到板子

```bash
# 给普通用户（装到 /home 下）
adb push ai-fitness-coach-installer.run /home/<用户名>/
adb shell chown <用户名>:<用户名> /home/<用户名>/ai-fitness-coach-installer.run

# 或给 root（装到 /root 下，无需 chown）
adb push ai-fitness-coach-installer.run /root/
```

### 3. 安装

```bash
# A. 给普通用户用（推荐）→ 装到 /home/<用户名>/ai-fitness-coach
adb shell su - <用户名> -c '/home/<用户名>/ai-fitness-coach-installer.run'

# B. 给 root 用 → 装到 /root/ai-fitness-coach（安装包推在哪就用哪的路径）
adb shell /root/ai-fitness-coach-installer.run
```

> **root 与普通用户怎么选**（板子上登录的用户不一定是 `q`，用 `<用户名>` 替换成真实用户名）：
>
> - **普通用户（推荐）**：装到家目录 `~/ai-fitness-coach`，在自己的桌面会话里双击运行。
> - **root**：装到 `/root/ai-fitness-coach`，**终端跑 `code/run.sh` 启动**——run.sh 会自动借用
>   当前活跃的桌面会话显示窗口，root 不登录桌面也能用（root 场景用 run.sh 就好，无需启动器）。
> - **不要 root 与普通用户交替运行**：root 实例会把日志目录属主变成 root，之后普通用户
>   的日志自动降级到 `/tmp/fitness-coach-<uid>.log`（不影响运行，只是日志换了位置）。

### 4. 启动

普通用户：文件管理器双击 `code/ai-fitness-coach`（重复双击 = 自动重启）。

root：终端跑

```bash
cd /root/ai-fitness-coach/code && ./run.sh
```

启动是否正常，看日志确认三个关键点：

```bash
tail -30 ~/.local/state/fitness-coach/app.log     # 日志目录不可写时自动降级 /tmp/fitness-coach-<uid>.log
# 确认: 1) [cam] opened idx=N @640x480   摄像头已打开
#       2) [pose] backend=qnn            NPU 双段推理链路就绪（首次初始化需几十秒）
#       3) [pose] ... fps=13~17          帧率正常
```

root 直跑 `run.sh` 时输出直接打印在终端，无需另查日志文件。

## 从源码构建（开发者可选）

> 普通用户直接用现成 `.run` 安装包即可，无需走本节。
> 打包在**开发主机（x86 Linux）**上进行；开发板只负责安装运行。

### 1. 检查前置条件

```bash
cd H1-AI-Fitness-Coach          # 后文所有相对路径以此为准
python3 --version               # ≥ 3.8 即可（跑构建脚本）
tar --version && xz --version   # 打包压缩用
curl --version                  # 下载 conda 包用
adb devices                     # 必须能看到板子（USB 连接）
```

- 板子侧零配置：出厂 Debian 13 桌面环境即可，无需预装任何依赖
- 双击启动器已预编译在仓库内（2 MB 静态 aarch64 ELF），打包**不需要 Go**；
  仅当改了 `tools/launcher/main.go` 才需重新交叉编译：
  `GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o code/ai-fitness-coach`

### 2. 构建 Qt 运行时（runtime/，仓库不含此目录，约 1 GB）

界面依赖 PySide6/Qt6（aarch64、Python 3.13）。依赖清单优先用仓库内置的 `pkgs.txt`
（已验证可用的版本组合）；仅在需要升级依赖时才重新生成：

```bash
tools/download_conda_pkgs.sh    # 按内置 pkgs.txt 从清华镜像下载 conda 包（可重复执行）
tools/download_wheels.sh        # 下载 4 个 aarch64 wheel（cv2/protobuf/flatbuffers/packaging）
tools/make_runtime.sh           # 解包合并 -> runtime/（py/ + lib/ + share/）
```

> **为什么优先用内置清单**：实时解析（`python3 tools/resolve_conda_deps.py > pkgs.txt`）
> 从镜像取「最新满足版本」，而 conda 镜像的 repodata 每天都在更新，可能拼出二进制不
> 兼容的组合（已实测踩坑：解析到 icu 79 而 PySide6 需要 libicui18n.so.78，应用启动即
> `ImportError`）。重新生成清单后构建的 runtime，建议先上板验证再替换内置清单。

已在板上装过本应用时，也可以直接复用安装目录里的 `runtime/`。

### 3. （可选）语音与模型再生成

仓库已内置全部产物（`audio/` 中英 128 条 wav、`models/` 双段模型），以下仅在需要重制时执行：

```bash
# 语音包：需 edge-tts + miniaudio（主机生成时联网）
pip install --target ./edge-tts-py edge-tts miniaudio
PYTHONPATH=./edge-tts-py python3 tools/gen_tts.py
```

### 4. 一键打包

```bash
tools/make_run_pack.sh
# 产物：dist/ai-fitness-coach-installer.run（约 222MB，自解压、root 与普通用户均可装）
```

构建完成后，按**快速开始第 2~4 步**推送、安装、启动。

> **日常改代码**：无需重新打包，直接把改动的 `code/*.py` 推到板上安装目录
> `~/ai-fitness-coach/code/` 即生效。改了 `models/`、`audio/`、`assets/` 或
> `runtime/` 才需要重新打包安装。

## 使用方法

1. 打开动作下拉框选择动作 — 每种动作可预览示范视频；切换有 1.5s 冷却 + 重复选择忽略（防运动时误触）
2. 按 **▶ 开始训练** — 启动后是**待机**状态（只预览不计次），按了才开始记录
3. 面对摄像头训练 — 自动计数并语音播报，未计次的动作会在界面显示原因
4. 底部按钮随时可用：暂停 / 继续、归零、数字变焦、中英切换
5. 训练结束按 **Esc** 退出（圆角窗口没有关闭按钮）

### 计次规则

| 动作 | 计一次的判定 |
|---|---|
| 深蹲 | 膝角低于约 115° 后回到 145° 以上，躯干前倾 < 55°（踝丢失时自动降级髋角/高度兜底） |
| 开合跳 | 双脚分开超过 1.35 倍肩宽且双腕过头顶，再回到并拢（启动需先观察到并拢姿势） |
| 高抬腿 | 任一膝盖抬到髋下 0.5 躯干长以内再落地还原，左右腿独立各计 1 |
| 弓步蹲 | 前弓腿膝角低于约 135°（持续 2 帧），上身保持挺直，蹲下再站直算一次 |
| 侧平举 | 双腕同时抬到肩下 0.15 躯干长以内，再双臂回落（滞回防抖） |
| 马步蹲 | 双脚超过 1.1 倍肩宽且膝角低于约 135° 开始计时，中途起身不清零，每 10/20/30/45/60s 语音里程碑 |

### 环境变量（run.sh 已内置默认值）

| 变量 | 默认 | 说明 |
|---|---|---|
| `CAMERA_INDEX` | 自动 | 运行时扫描 /dev/video* 自动发现；设数字可强制指定 |
| `POSE_BACKEND` | qnn | `qnn`（NPU+CPU 双段 ~43ms）/ `onnx`（CPU 整模型 ~100ms 兜底） |
| `FITNESS_LANG` | zh | 初始语言 `zh` / `en` |
| `FITNESS_SNAPSHOT` | - | 调试：设路径则启动后自动截图 UI（`FITNESS_SNAPSHOT_MS` 调延迟，默认 15s） |
| `FITNESS_RESTART_DELAY` | 30 | 杀掉旧实例后等待 NPU（CDSP）复位的秒数，0 为不等待 |

## 常见问题

**检测不到摄像头？**

在板上执行 `ls /dev/video*` 确认设备（应用实测用 video0，会自动扫描发现）；重新插拔 USB 摄像头，插入后等待几秒再启动应用。摄像头是独占资源，应用内有打开失败 15 次重试兜底。

**动作不计数？**

应用只计标准动作。查看界面上的未计次原因（如蹲得不够深），离摄像头 2~4 米、正身、光线充足再试。

**没有语音？**

应用跟随系统音频输出路由（pw-play 优先，失败回退 aplay）。检查系统设置的输出设备。

**启动就退出 / 反复起不来？**

不要连续快速重启：持有 QNN 会话的实例退出后，NPU（CDSP）需要约半分钟复位，紧跟着启动的新实例会在模型加载阶段静默退出。启动器/脚本已内置 30 秒复位等待（`FITNESS_RESTART_DELAY` 可调）；若仍起不来，停掉残留实例（`pkill -f "[m]ain.py"`，严禁 `kill -9`）等一分钟再启动即可。

## 技术说明

模型双段拆分与转换、QNN 运行时接入、运行时打包与完整防误计参数，详见 [TECH.md](TECH.md)。
