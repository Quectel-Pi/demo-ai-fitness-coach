# 智能健身教练 · 技术资料

> 本文档是项目的技术参考手册：完整技术栈、选型依据、实现原理、版本信息和官方资料链接。
> 使用说明（安装/操作/站位要求）见 [README.md](README.md)。
> 最后更新：2026-10-15

---

## 目录

1. [项目概览与核心指标](#1-项目概览与核心指标)
2. [硬件平台](#2-硬件平台)
3. [系统与图形音频栈](#3-系统与图形音频栈)
4. [AI 推理链路（核心）](#4-ai-推理链路核心)
5. [动作分析算法](#5-动作分析算法)
6. [UI 与交互（PySide6）](#6-ui-与交互pyside6)
7. [语音链路](#7-语音链路)
8. [部署与分发工程](#8-部署与分发工程)
9. [关键设计决策与踩坑记录](#9-关键设计决策与踩坑记录)
10. [参考资料汇总](#10-参考资料汇总)

---

## 1. 项目概览与核心指标

USB 摄像头实时识别 17 个人体关键点（YOLO11n-pose，RK3576 NPU 推理），
对深蹲/开合跳/高抬腿/弓步蹲/侧平举/马步蹲六种动作做**计次/计时 + 实时纠错 + TTS 语音**，
PySide6 漫画风 UI，板端完全离线运行。

| 指标 | 数值 | 说明 |
|---|---|---|
| NPU 推理耗时 | **~55 ms/帧（≈18 FPS）** | YOLO11n-pose 640x640 fp16 |
| CPU 兜底推理 | ~295 ms/帧 | onnxruntime aarch64，4 线程 |
| 加速比 | **5.4x** | NPU vs CPU |
| 整链路 UI 帧率 | ~14 FPS | 含采集/后处理/绘制 |
| 语音资源 | 64 条 x 2 语言 | zh 晓晓 / en Jenny，wav 离线播放 |
| 一键安装包 | ~215 MB | .run 自解压（tar.xz） |
| 板上占用 | ~1 GB | runtime（conda PySide6 全家桶 985MB） |
| 模型文件 | ~6 MB | yolo11n-pose.rknn（fp16） |

---

## 2. 硬件平台

### 2.1 RK3576 SoC

| 项目 | 规格 |
|---|---|
| CPU | 4x Cortex-A72 + 4x Cortex-A53（big.LITTLE，最高 2.2GHz） |
| GPU | ARM Mali-G52 MC3（OpenGL ES / Vulkan） |
| **NPU** | **6 TOPS 自研 RKNN 架构**，支持 INT4/INT8/INT16/FP16/BF16 |
| 视频 | 8K30 解码（H.265/VP9/AV1）、4K60 编码 |
| 显示 | DisplayPort / MIPI / eDP / HDMI / RGB / EBC，三屏异显 |
| 内存 | 32bit LPDDR4/4x、LPDDR5 |

- 瑞芯微 RK3576 官方产品页：<https://www.rock-chips.com/a/cn/product/RK35xilie/2024/1015/2016.html>
- 瑞芯微官网：<https://www.rock-chips.com/>
- RK3576 已进入 Linux 主线内核（6.12 起，Collabora 推动上游化）

### 2.2 开发板与外设

| 硬件 | 型号/规格 | 用途 |
|---|---|---|
| 开发板 | **QUECPI M2**（移远 Quectel，RK3576 核心板） | 主控 |
| USB 摄像头 | 640x480 YUYV 定焦 | 姿态输入（V4L2 自动发现，USB 优先实测出图） |
| 显示屏 | MIPI 触摸屏（默认）/ HDMI 屏 | UI 显示 |
| 音频输出 | 板载 **ES8389** codec（模拟口）+ 蓝牙（AIC 组合芯） | 语音播放 |
| 连接 | USB 数据线 adb / WiFi / 蓝牙 | 部署调试 |

- Quectel 官网：<https://www.quectel.com/>
- V4L2 用户态 API 文档：<https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/v4l2.html>

---

## 3. 系统与图形音频栈

| 层 | 组件 | 说明 |
|---|---|---|
| OS | **Debian 13**（aarch64） | <https://www.debian.org/> |
| 桌面 | **GNOME** | <https://www.gnome.org/> |
| 显示协议 | **Wayland**（无 X11） | <https://wayland.freedesktop.org/> |
| 音频服务 | **PipeWire**（兼容 PulseAudio 客户端） | <https://pipewire.org/> |
| Python | **3.13**（板上系统自带） | <https://www.python.org/> |
| 调试 | adb（Android Debug Bridge） | <https://developer.android.com/tools/adb> |

要点：

- GUI 程序需要 Wayland 会话环境（`XDG_RUNTIME_DIR` + `WAYLAND_DISPLAY` + `DBUS_SESSION_BUS_ADDRESS`）；
  run.sh 运行时自动扫描 `/run/user/*/wayland-*`，root 会话下也可跨用户连 q 的 GNOME 会话
- 音频走 PipeWire 系统路由：`paplay`（PulseAudio 用户工具）优先——跟随用户选择的输出设备
  （蓝牙耳机可用），失败回退 `aplay`（ALSA 直出 ES8389）
  - PulseAudio 用户文档：<https://www.freedesktop.org/wiki/Software/PulseAudio/Documentation/User/>
  - ALSA 项目：<https://www.alsa-project.org/wiki/>

---

## 4. AI 推理链路（核心）

### 4.1 模型选型：为什么是 YOLO11n-pose

#### 4.1.1 选型约束（需求倒推）

选型不是从"哪个模型最准"开始，而是从板子的硬约束开始：

| # | 约束 | 推论 |
|---|---|---|
| 1 | 推理必须跑在 RK3576 的 6 TOPS NPU 上（CPU 推理 ~300ms 不可用） | 模型必须能经 ONNX→RKNN 工具链转换 |
| 2 | 板上 Python 3.13 aarch64、无外网 | 依赖只能用离线可装的包/wheel |
| 3 | 单人健身场景 | 只需 1 人 × 17 关键点，不需要多人、手势、人脸 |
| 4 | 实时性：推理 ≥15 FPS | 640×640 输入下需 ~60ms 内完成 |
| 5 | 动作判定角度阈值灵敏度 ±5~10° | 关键点坐标必须稳定，对量化敏感 |

#### 4.1.2 候选方案全对比

| 方案 | 关键点格式 | RKNN/NPU 可部署性 | 调研结论 |
|---|---|---|---|
| MediaPipe Pose（BlazePose） | 33 点自定义 | PyPI 无 aarch64 Linux wheel；源码编译需 Bazel + Android 工具链，板上不可行 | **弃用**（部署性一票否决） |
| OpenPose（CMU） | COCO-18 / BODY_25 | 多阶段重量级网络，无官方端侧支持，CPU 实时不可行 | 弃用 |
| MoveNet（Google） | 17 点（COCO） | TFLite 生态，无官方 TFLite→RKNN 转换链路 | 弃用 |
| RTMPose（MMPose） | COCO-17 | 可转 ONNX，但依赖 mmcv/mmdeploy 全家桶，RKNN 社区资料少、链路重 | 备选，未采用 |
| YOLOv8-pose（Ultralytics） | COCO-17 | 官方支持 RKNN 导出；瑞芯微 rknn_model_zoo 收录官方 yolov8_pose 示例 | 可行备选 |
| **YOLO11n-pose（Ultralytics）** | COCO-17 | 与 v8 完全相同的导出工具链；新一代架构 | **采用** |

#### 4.1.3 决定性理由

1. **部署性一票否决制**。MediaPipe 精度口碑好、免费、CPU 友好，但 aarch64 产物缺失直接出局。
   本项目第一个实测教训：**端侧选型第一步先查目标架构有没有现成产物
   （wheel/模型/转换案例），再谈精度**——模型再准，进不了 NPU 都是零分。
2. **双向生态验证**。Ultralytics 官方文档明确支持 Rockchip RKNN 导出并列出 RK3576；
   瑞芯微官方 rknn_model_zoo 收录同架构家族的 yolov8_pose 完整示例——
   opset 兼容、量化行为、后处理格式两边都有人趟过路，踩坑面最小。
3. **单阶段（one-stage）架构天生适合端侧**：找人和回归关键点在**同一次前向**完成，
   不需要"先检测人→裁剪→再跑 pose"的两阶段流水线——延迟减半，工程链路减半。
4. **回归式（regression-based）输出对 NPU 友好**：YOLO-pose 每个关键点直接回归 (x, y, conf)；
   而 MediaPipe/RTMPose 等热图式（heatmap-based）方法输出 17×H×W 概率热图，
   还要 CPU 做 argmax + 亚像素细化。前者输出即坐标，板端后处理几乎为零。
5. **COCO 17 关键点是行业最大公约数**：本项目实际只用 12 个
   （鼻/肩×2/肘×2/髋×2/膝×2/踝×2，跳过眼耳 5 点），多余点对单人健身是冗余计算；
   但标准 COCO 格式意味着将来换模型/数据集零迁移成本。MediaPipe 33 点同理冗余且非标准。
6. **YOLO11 vs YOLOv8**：同一厂商、同一导出工具链、同一 RKNN 支持（pose 任务均为
   FP16-only），YOLO11（2024 年发布）是新一代：官方口径同等精度下参数更少/推理更快，
   骨干尾部新增空间注意力模块（C2PSA）对关键点定位直接受益。部署成本相同，选新一代。

#### 4.1.4 规模档位：为什么是 n（nano）

YOLO11-pose 有 n/s/m/l/x 五档。选 n 的依据：

- **场景精度余量**：单人、1.5~4m、正身——目标大且清晰，n 档定位精度实测可满足
  角度状态机 ±5° 的灵敏度要求
- **实测性能**：640×640 fp16 在 NPU 上 ~55ms（≈18 FPS），给 CPU 前后处理和 UI 留足预算
- **体积**：rknn 产物仅 ~6MB，对 215MB 安装包无压力
- **升级路径**：将来若需多人/远距场景，换 s 档只是换一个模型文件，代码不动

参考链接：

- Ultralytics 官网/文档：<https://docs.ultralytics.com/>，GitHub：<https://github.com/ultralytics/ultralytics>
- YOLO11 模型资产下载：<https://github.com/ultralytics/assets>
- Pose 任务文档（各档位精度/速度官方对比）：<https://docs.ultralytics.com/tasks/pose/>
- COCO 17 关键点定义（论文 *Microsoft COCO: Common Objects in Context*）：<https://arxiv.org/abs/1405.0312>
- MediaPipe（弃用方案参考）：<https://github.com/google-ai-edge/mediapipe>
- MoveNet（弃用方案参考）：<https://www.tensorflow.org/hub/tutorials/movenet>
- RTMPose（备选参考）：<https://github.com/open-mmlab/mmpose>

### 4.2 YOLO11-pose 网络结构一瞥

理解网络结构才能理解后处理为什么那样写。单阶段三段式：

```
输入 640×640×3（letterbox 后的 RGB）
   │
   ▼ Backbone（特征提取）
   Conv 下采样 → C3K2 模块堆叠（v8 时代 C2F 的高效演进）→ SPPF（快速空间金字塔池化）
   → C3K2 → C2PSA（空间注意力，增强关键点定位）
   │
   ▼ Neck（多尺度融合，PAN-FPN 双向）
   P3（80×80 小目标）/ P4（40×40）/ P5（20×20 大目标）三尺度双向融合
   │
   ▼ Head（解耦头，anchor-free）
   框回归分支(4) + 目标置信度分支(1) + 关键点回归分支(17×3)
   │
   ▼ 输出 (1, 56, 8400)
```

**输出张量解读**（[code/pose_estimator.py](code/pose_estimator.py) 后处理的依据）：

- **8400 = 80² + 40² + 20²**：三个尺度网格上的预测位置数（anchor-free，
  每个网格位置直接出一套完整预测）
- **56 = 4 + 1 + 17×3**：框 (x,y,w,h) + 是人的置信度 + 17 个关键点各自的 (x, y, conf)
- 每个网格位置都在"猜"一套完整答案，后处理负责挑出最可信的那一个

**单人场景的工程简化**：标准 YOLO 后处理要做 NMS（非极大值抑制）去重多个候选框；
本项目单人健身，直接取目标置信度最大的网格位置——省掉 NMS 计算，也天然免疫
"背景里出现第二个人/海报人像"的干扰。这是场景驱动的简化；若做多人场景需恢复
NMS 并按框分配关键点。

**坐标还原（letterbox 逆变换）**：输入前做了等比缩放 + 居中填充（114 灰），
推理输出的关键点坐标需 `(x - pad_x) / scale` 逆变换回原图像素坐标。

### 4.3 转换流水线（主机端，x86 Linux）

```
yolo11n-pose.pt (Ultralytics)
  └─导出→ yolo11n-pose.onnx        (opset 22，原始)
       └─降 opset→ ...-op19.onnx   (opset 19 —— rknn-toolkit2 支持上限)
            └─rknn-toolkit2 转换→ yolo11n-pose.rknn  (fp16, target=rk3576)
```

转换配置（[tools/convert_rknn.py](tools/convert_rknn.py)）：

- `mean_values=[[0,0,0]], std_values=[[255,255,255]]` —— 对应 Ultralytics 的 `x/255` 归一化，
  归一化放进模型，推理端只喂 uint8→float32
- `target_platform="rk3576"`
- **`do_quantization=False`（fp16 不量化）** —— 见 4.5 精度决策

工具链版本（关键，升级需回归测试）：

| 组件 | 版本 | 备注 |
|---|---|---|
| rknn-toolkit2 | **2.3.2** | x86 cp310 wheel，pip 装进隔离目录 `.build/rknn-env`，不污染宿主机 |
| onnx | 1.15.0 | toolkit2 2.3.2 兼容版 |
| protobuf | 4.25.3 | 同上 |

官方资料：

- rknn-toolkit2（模型转换/仿真/量化，含板端 Runtime 说明）：<https://github.com/airockchip/rknn-toolkit2>
- 官方文档目录：<https://github.com/airockchip/rknn-toolkit2/tree/master/doc>
- RKNN Model Zoo（各任务官方转换示例与预训练模型）：<https://github.com/airockchip/rknn_model_zoo>
- Ultralytics 官方 Rockchip RKNN 部署指南（明确列出 RK3576 支持；
  官方口径：**FP16 支持全部任务，INT8 目前仅检测模型**——与本项目 fp16 决策一致）：
  <https://docs.ultralytics.com/integrations/rockchip-rknn/>

### 4.4 板端推理：自写 ctypes 绑定

板上 **Python 3.13 没有 rknnlite（RKNN-Toolkit-Lite2）的 cp313 wheel**，
因此 [code/rknn_backend.py](code/rknn_backend.py) 用 `ctypes` 直接绑定
`librknnrt.so` 的 **C API**（结构体照 `rknn_api.h` 手写）：

- `rknn_init` / `rknn_query`（IO 数量、tensor 属性）/ `rknn_inputs_set` / `rknn_run` / `rknn_outputs_get`
- 输入 **NHWC float32**，NPU runtime 自动做量化进芯片
- 输出 `want_float=1` 取回 float32，免手动反量化

后处理（[code/pose_estimator.py](code/pose_estimator.py)）：

- 模型输出 `(56, 8400)`：4 框坐标 + 1 置信度 + **17 关键点 x 3**（x, y, conf）
- 单人场景：直接取置信度最大的 anchor（省 NMS）
- letterbox 逆变换还原到原图像素坐标，返回 `(17, 3)` 关键点数组
- 双后端抽象：`POSE_BACKEND=rknn|onnx`（onnxruntime CPU 兜底，~300ms）

### 4.5 精度决策：fp16 而非 int8

| 对比项 | fp16 | int8 量化 |
|---|---|---|
| mAP 损失 | 基本 <0.3%（近无损） | 小模型（YOLO11n 级）可能掉 2~3 个点 |
| 关键点坐标 | 稳定 | 抖动 → 角度状态机误判风险 |
| NPU 速度 | 基准 | 快 ~1.4x |
| 端到端收益 | — | 仅 10~20%（瓶颈在 CPU 前后处理与摄像头采集） |

结论：**动作判定的角度阈值（±5~10° 灵敏度）对关键点坐标精度敏感**，
int8 的坐标抖动会直接影响状态机稳定性，而速度收益被前后处理瓶颈吃掉——选 fp16。

### 4.6 许可证注意

- Ultralytics YOLO11：**AGPL-3.0**（商用需评估，内部演示/学习无碍）
  <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>
- 本项目为演示用途；正式分发前请按上游许可证逐一核查

---

## 5. 动作分析算法

实现：[code/exercise_analyzer.py](code/exercise_analyzer.py)。纯 numpy 几何计算，不依赖额外模型。

### 5.1 数据流

```
(17,3) 关键点 → 几何特征（三点夹角/比例/高度）→ 状态机 → Feedback(event, count, stage, problems)
```

关键点索引（COCO）：鼻0 / 肩5-6 / 肘7-8 / 腕9-10 / 髋11-12 / 膝13-14 / 踝15-16

### 5.2 设计哲学：几何规则状态机 vs 端到端动作分类

拿到每帧 17 个关键点后，"判断这是不是一个合格的深蹲"存在两条技术路线，
这是本项目算法层最重要的架构决策：

| 路线 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A. 端到端时序分类**（ST-GCN / 时序 Transformer 等） | 把关键点序列（骨骼序列）喂给图卷积/时序网络，直接输出动作类别 + 质量分 | 理论上限高；无需手工设计规则 | 每个动作需数百段标注视频训练；黑盒不可解释；改判定标准 = 重新采集标注训练；给不出「这次蹲到 121°，差 6°」这种量化反馈 |
| **B. 几何规则状态机**（本项目） | 关键点 → 三点夹角/长度比例 → 阈值状态机（见 5.4） | **可解释**：每个判定都有角度证据链；**零训练数据**；阈值即业务规则，随时可调；角度经躯干长归一化后跨体型/跨距离泛化 | 规则需运动学知识设计；遮挡/出镜等边界情况要手工兜底（已用三层兜底解决） |

**选 B 的决定性理由——健身教练的核心价值是纠错反馈，不是分类**：

- 用户要的不只是"深蹲 ×1"，而是「上次蹲不够深(121°) 未计次」「膝盖不要内扣」——
  规则路线天然输出这种**量化、可行动**的反馈；端到端分类只能给一个类别标签
- 「普通人标准 vs 体育级标准」在规则路线里就是两个阈值数字（膝角 115° vs 90°），
  改一行代码；分类模型要重新采集两种标准的训练数据
- 角度是**人体测量学不变量**：无论用户高矮胖瘦、离摄像头远近（比例归一化后），
  一个标准深蹲的膝角都在 ~115° 附近——泛化能力免费获得，无需多样性训练集
- 板上零额外开销：纯 numpy 几何计算，不占 NPU，推理预算全留给视觉模型

路线 B 的短板（关键点丢失时规则失效）由 5.3 的防误计机制和深蹲三层兜底（5.4）补齐——
本质是**用工程复杂度换可解释性**，对健身纠错场景是正确的交换。

### 5.3 通用防误计机制（跨动作复用）

| 机制 | 说明 |
|---|---|
| **帧确认（去抖）** | 状态切换需连续 N 帧证据（如深蹲入蹲 3 帧/起身 2 帧），单帧噪声不翻转状态 |
| **三态判定** | 动作成立=True / 不成立=False / **无法判断=None**；None 时清票不当作证据，防止跨丢帧拼凑计次 |
| **测量点过滤** | 参与计算的关键点要求 conf≥0.30 + y 坐标解剖学范围校验（如膝盖不可能高于肩膀），静立噪声不进状态机 |
| **动作级阈值** | 每动作可覆盖全局阈值（如高抬腿专属关键点置信度 0.15，其余动作 0.25） |
| **弯腰门闸** | 躯干前倾 >55° 时禁止进入下蹲类状态（弯腰≠蹲） |
| **比例归一化** | 所有距离用躯干长/肩宽归一化，与人和摄像头的距离无关 |

### 5.4 各动作判定阈值表

| 动作 | 核心特征 | 关键阈值 | 计次条件 |
|---|---|---|---|
| **深蹲** | 左右膝平均角度 | 蹲到位 115° / 站立 145° / 入蹲 125° | 完整蹲起且最低膝角 <115° |
| **开合跳** | 脚距/肩宽 + 腕高 | 打开 1.35x肩宽+腕过鼻尖 / 并拢 0.6x | 完整"并拢→打开"，启动需先观察到并拢（3 帧就位确认） |
| **高抬腿** | 膝高/躯干长（左右腿独立状态机） | 抬腿 0.5x躯干长 + 踝差校验 | 抬起→落地，左右各计 1 |
| **弓步蹲** | 双膝角差 + 弯曲腿角度 | 弓步 任一膝<140°或角差>25° / 站立 >150°且角差<12° | 弯曲腿最低 <135° |
| **侧平举** | 腕高/躯干长 | 举起 肩下 0.15x / 放下 肩下 0.4x（滞回） | 举起→放下，双臂同步 |
| **马步蹲（计时）** | 平均膝角 + 踝距 | 保持 <135° 且踝距 >1.1x肩宽 | 累计秒数，中断不损失 |

**深蹲三层兜底**（适应遮挡/出镜）：

1. 膝角模式（ ankle 可见）：角度状态机
2. 髋角兜底（踝丢失）：肩-髋-膝角度，阈值同步放宽
3. 高度兜底（膝也丢失，肩+髋可见即可）：髋部相对站立基线下沉量
   （入闸 0.65x 躯干长 / 出闸 0.22 / 低位滞留 4s 重置基线 / 连续 2 帧上半身可见才重锚）

纠错提示：膝内扣（膝盖相对髋-踝连线向中线偏移 >9%，左右分开取 max 防符号抵消）、
背部前倾、手臂未过顶、双脚不够开、膝盖不够高等，均带 6s 冷却防刷屏。

### 5.5 已知原理性限制

高度兜底模式无法区分"真深蹲"与"蹲低看屏幕"（走近摄像头蹲下可能 +1）——
运动学上髋部下沉量是同一信号。已用**训练门闸**（待机/暂停不计次）从交互层规避。

---

## 6. UI 与交互（PySide6）

- 框架：**PySide6（Qt for Python 6）**，LGPL-3.0
  - 文档：<https://doc.qt.io/qtforpython-6/>，Qt：<https://www.qt.io/>
- 无框圆角窗口：26px 弧形圆角 + 蓝紫渐变底（#74C0FC→#B197FC）+ 4px 漫画描边；
  Wayland 下窗口尺寸运行时自适应屏幕
- **镜像预览**（符合自拍直觉）+ **数字变焦 0.5~3x**（放大=中心裁剪后送推理，远距识别更稳）
- 线程模型：摄像头采集 + NPU 推理在工作线程，UI 主线程只做绘制与交互（帧队列有界防积压）
- **训练门闸状态机**：待机（只预览）→ 训练中 → 暂停（冻结保留）→ 继续；归零按钮清计数/计时
- 触摸屏防误触：动作切换 1.5s 冷却 + 重复选择忽略（运动中手掌扫屏不误切）

---

## 7. 语音链路

### 7.1 合成（主机端，一次性）

- **edge-tts**（微软 Edge 浏览器在线 TTS 服务的 Python 客户端）
  - GitHub：<https://github.com/rany2/edge-tts>，PyPI：<https://pypi.org/project/edge-tts/>
- 音色与调参（[tools/gen_tts.py](tools/gen_tts.py)）：
  - 中文：`zh-CN-XiaoxiaoNeural`（晓晓，热情女声），语速 +15%、音调 +8Hz
  - 英文：`en-US-JennyNeural`（Jenny），+10%、+4Hz
- 产物：`audio/zh`、`audio/en` 各 64 条 wav，**板上完全离线播放**
- 微软神经语音音色列表：<https://learn.microsoft.com/en-us/azure/ai-services/speech-services/language-support#text-to-speech>
- 曾用方案 piper（chaowen 音色）因 espeak 口音淘汰——经验：中文 TTS 务必试听真实输出

### 7.2 播放（板端）

```
audio_player.py 播放队列（串行，防重叠）
  ├─ paplay  → PipeWire 系统路由（跟随用户输出设备，蓝牙耳机可用）  ← 优先
  └─ aplay   → ALSA 直出 ES8389（root 会话 PipeWire 拒连时兜底）
```

- 播放前探测：`pactl info` 可用则走 paplay，否则 aplay

---

## 8. 部署与分发工程

### 8.1 离线依赖（板上无外网）

- PySide6/Qt6 全家桶来自 **conda-forge aarch64 包**（94 个，~985MB），
  用 [tools/resolve_conda_deps.py](tools/resolve_conda_deps.py) 解析依赖闭包 +
  [tools/download_conda_pkgs.sh](tools/download_conda_pkgs.sh) 清华镜像批量下载 +
  [tools/make_runtime.sh](tools/make_runtime.sh) 解压合并出 `runtime/`（PYTHONPATH + LD_LIBRARY_PATH）
- conda-forge：<https://conda-forge.org/>
- onnxruntime/protobuf/flatbuffers 等用 PyPI aarch64 wheel
- 精简空间可删 `libLLVM/libclang`（~340MB，仅 QtUiTools 用）

### 8.2 .run 自解压安装包（[tools/make_run_pack.sh](tools/make_run_pack.sh)）

```
installer.run = bash header（自解压脚本）+ tar.xz payload（code/models/audio/runtime/README）
```

- 板上**最小系统无 xz 命令** → payload 用 python3 内置 lzma 流式解压
  （`tarfile.open(mode="r|xz")`）
- 解压脚本必须 `python3 -c` 传参——**heredoc（`python3 - <<EOF`）会抢占 stdin**，
  导致管道里的 payload 读不到（实测踩坑）
- 安装包 root 与普通用户均可执行；装到 /home 下时 root 自动 chown 回属主

### 8.3 双击启动器（Go 静态交叉编译）

- GNOME 新版 nautilus 双击 .sh 会弹「运行/显示」对话框，**双击 ELF 二进制直接执行**（AppImage 原理）
- [tools/launcher/main.go](tools/launcher/main.go)：
  `GOOS=linux GOARCH=arm64 CGO_ENABLED=0` 交叉编译出 1.4MB **静态 aarch64 ELF**
- 职责：杀旧实例 → 补齐 Wayland 环境变量 → `setsid` 拉起 run.sh（完全脱离会话）→ 日志写
  `~/.local/state/fitness-coach/app.log`
- Go：<https://go.dev/>

### 8.4 板上常用命令（adb）

```bash
# 热更新单文件并重启
adb push code/xxx.py /home/q/ai-fitness-coach/code/ && \
adb shell "chown q:q /home/q/ai-fitness-coach/code/xxx.py; pkill -f '[m]ain.py'; \
  su - q -c 'setsid /home/q/ai-fitness-coach/code/run.sh >> /home/q/.local/state/fitness-coach/app.log 2>&1 < /dev/null &'"
```

注意：板上 `runuser` 已损坏，用 `su - q -c` 替代；`pkill -f '[m]ain.py'` 加方括号防自杀。

---

## 9. 关键设计决策与踩坑记录

| # | 问题 | 决策/教训 |
|---|---|---|
| 1 | MediaPipe 无 aarch64 wheel | 换 YOLO11n-pose；端侧选型先查目标架构产物 |
| 2 | 板上无 rknnlite cp313 包 | ctypes 直调 librknnrt.so C API，一次绑定永久可用 |
| 3 | int8 提速仅 10~20% 且抖关键点 | fp16 不量化；量化决策要算端到端账，不只看 NPU 带宽 |
| 4 | opset 22 ONNX 转换失败 | rknn-toolkit2 上限 opset 19，导出时降 opset |
| 5 | 板上无 xz 命令 | python3 lzma 流式解压；且不能用 heredoc 抢 stdin |
| 6 | 双击 .sh 弹对话框 | Go 静态 ELF 做启动入口（AppImage 原理） |
| 7 | root 会话 PipeWire 拒连 | paplay 优先 + aplay 直出 ES8389 兜底的双链路 |
| 8 | dconf/gsettings 写入失败 | 需 XDG_RUNTIME_DIR 属于当前用户（root 的 /var/run 不行） |
| 9 | 弯腰/走动/躺卧误计 | 帧确认 + 三态清票 + 弯腰门闸 + 测量点过滤 + 训练门闸分层拦截 |
| 10 | numpy bool 陷阱 | `np.bool_(False) is False` 为 False！状态机判定函数必须 `return bool(x)` |
| 11 | 「未检测到人」时提前返回丢计数 | 返回值必须携带当前 count，否则 UI 计数 0/1 抖动 |
| 12 | 膝内扣左右抵消 | 左右腿分开判方向（朝身体中线才算），取 max 而非平均 |
| 13 | 高抬腿静立自增 | 测量点 conf + 解剖学范围双重过滤 |
| 14 | 蓝牙耳机断续 | 排查顺序：sink 状态 → dmesg 看 AIC 组合芯掉电循环 → 换设备隔离 |
| 15 | pkill 杀掉 adb 会话 | 模式加方括号 `[m]ain.py`；重启用 setsid 脱离会话 |

---

## 10. 参考资料汇总

### 硬件 / SoC

| 资料 | 链接 |
|---|---|
| RK3576 官方产品页（瑞芯微） | <https://www.rock-chips.com/a/cn/product/RK35xilie/2024/1015/2016.html> |
| 瑞芯微官网 | <https://www.rock-chips.com/> |
| Quectel（QUECPI M2 厂商） | <https://www.quectel.com/> |
| V4L2 内核文档 | <https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/v4l2.html> |

### AI 框架与工具链

| 资料 | 链接 |
|---|---|
| rknn-toolkit2（模型转换/仿真/量化） | <https://github.com/airockchip/rknn-toolkit2> |
| rknn-toolkit2 官方文档目录 | <https://github.com/airockchip/rknn-toolkit2/tree/master/doc> |
| RKNN Model Zoo（官方示例模型库） | <https://github.com/airockchip/rknn_model_zoo> |
| Ultralytics Rockchip RKNN 部署指南 | <https://docs.ultralytics.com/integrations/rockchip-rknn/> |
| ONNX | <https://onnx.ai/> |
| onnxruntime（CPU 兜底后端） | <https://onnxruntime.ai/> |
| NumPy | <https://numpy.org/> |
| OpenCV（采集/绘制） | <https://docs.opencv.org/> |

### 模型与数据集

| 资料 | 链接 |
|---|---|
| Ultralytics YOLO11 文档 | <https://docs.ultralytics.com/> |
| YOLO11 模型资产（yolo11n-pose.pt） | <https://github.com/ultralytics/assets> |
| Pose 任务文档 | <https://docs.ultralytics.com/tasks/pose/> |
| COCO 关键点论文（17 点定义） | <https://arxiv.org/abs/1405.0312> |
| MediaPipe（弃用方案参考） | <https://github.com/google-ai-edge/mediapipe> |

### UI / 图形 / 系统

| 资料 | 链接 |
|---|---|
| PySide6（Qt for Python 6）文档 | <https://doc.qt.io/qtforpython-6/> |
| Qt 官网 | <https://www.qt.io/> |
| Wayland | <https://wayland.freedesktop.org/> |
| GNOME | <https://www.gnome.org/> |
| Debian | <https://www.debian.org/> |

### 语音 / 音频

| 资料 | 链接 |
|---|---|
| edge-tts（GitHub） | <https://github.com/rany2/edge-tts> |
| edge-tts（PyPI） | <https://pypi.org/project/edge-tts/> |
| 微软神经语音音色列表 | <https://learn.microsoft.com/en-us/azure/ai-services/speech-services/language-support#text-to-speech> |
| PipeWire | <https://pipewire.org/> |
| PulseAudio 用户文档（paplay） | <https://www.freedesktop.org/wiki/Software/PulseAudio/Documentation/User/> |
| ALSA（aplay） | <https://www.alsa-project.org/wiki/> |

### 部署 / 打包

| 资料 | 链接 |
|---|---|
| conda-forge（aarch64 离线包源） | <https://conda-forge.org/> |
| Go（启动器交叉编译） | <https://go.dev/> |
| adb（Android Debug Bridge） | <https://developer.android.com/tools/adb> |
| GNU tar | <https://www.gnu.org/software/tar/> |
| XZ Utils | <https://tukaani.org/xz/> |
| Python | <https://www.python.org/> |
