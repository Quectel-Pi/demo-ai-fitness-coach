# 智能健身教练（demo-ai-fitness-coach）

基于移远 Quectel Pi 系列开发板的边缘 AI 健身应用：通过 USB 摄像头实时识别 **17 个人体关键点**，对**深蹲 / 开合跳 / 高抬腿 / 弓步蹲 / 侧平举 / 马步蹲**六种动作自动**计次（计时）**，配合**实时动作纠错**与**中英双语语音播报**——全程离线运行，画面与数据不离开设备。

## 平台版本

同一套应用，按开发板 NPU 架构分别适配，选择你的板子进入对应目录：

| 目录 | 开发板 | AI 推理 | 安装包 |
|---|---|---|---|
| [QUECPI_M2/](QUECPI_M2/) | Quectel Pi M2（RK3576，6 TOPS NPU） | YOLO11n-pose → RKNN（fp16 不量化），单帧约 55ms | [Releases · v1.0.0-M2](https://github.com/Quectel-Pi/demo-ai-fitness-coach/releases/tag/v1.0.0-M2) |
| [QUECPI_H1/](QUECPI_H1/) | QuecPi H1（高通 QCS6490，Hexagon NPU HTP v68） | YOLO11n-pose 双段推理（主干 QDQ int8 走 NPU 约 6.6ms + 检测头 fp32 走 CPU 约 36ms），单帧约 43ms | [Releases · v1.0.0-H1](https://github.com/Quectel-Pi/demo-ai-fitness-coach/releases/tag/v1.0.0-H1) |

## 功能特性（两平台共有）

- **六种动作**：深蹲 / 开合跳 / 高抬腿 / 弓步蹲 / 侧平举（计次）+ 马步蹲（计时）
- **防误计门闸**：多帧确认 + 深度 / 躯干阈值；动作未计次时界面上显示具体原因（如「上次蹲不够深(121°) 未计次」）
- **训练控制**：待机门闸（点「开始训练」前只预览不计数）、暂停冻结成绩、一键归零
- **中英双语**：界面、提示、语音一键切换，离线语音包随安装包内置
- **数字变焦** 0.5×–3×（步进 0.25），内置六种动作标准示范视频
- **一键部署**：自解压安装包内置全部运行环境，板上零依赖、无需联网；安装后文件管理器双击即用

## 快速开始

1. 从对应平台的 [Releases](https://github.com/Quectel-Pi/demo-ai-fitness-coach/releases) 页下载安装包 `ai-fitness-coach-installer.run`
2. 按所选平台目录内的 README 部署（两条 adb 命令 + 双击启动）：

   - M2 平台：[QUECPI_M2/README.zh-CN.md](QUECPI_M2/README.zh-CN.md)
   - H1 平台：[QUECPI_H1/README.md](QUECPI_H1/README.md)

各平台的模型转换、推理链路与完整防误计参数见对应目录 `TECH.md`。
