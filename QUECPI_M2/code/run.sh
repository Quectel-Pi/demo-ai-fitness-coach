#!/bin/bash
# 板上启动脚本：自动发现 摄像头 / 桌面会话，不写死任何节点号
set -u
DIR="$(cd "$(dirname "$0")/.." && pwd)"

export LD_LIBRARY_PATH="$DIR/runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$DIR/code:$DIR/runtime/py"
export QT_PLUGIN_PATH="$DIR/runtime/lib/qt6/plugins"

# ---- 桌面会话定位 ----
# 普通用户双击启动: 自己的会话环境天然存在（XDG_RUNTIME_DIR + WAYLAND_DISPLAY 都有），
# 直接用不扫描; adb root 等无会话环境才扫描 /run/user/*/wayland-* 借用户会话
if [ -z "${WAYLAND_DISPLAY:-}" ] || [ -z "${XDG_RUNTIME_DIR:-}" ]; then
    for udir in /run/user/*; do
        [ -d "$udir" ] || continue
        for wl in "$udir"/wayland-*; do
            [ -S "$wl" ] || continue
            export XDG_RUNTIME_DIR="$udir"
            export WAYLAND_DISPLAY="${wl#"$udir"/}"
            export DBUS_SESSION_BUS_ADDRESS="unix:path=$udir/bus"
            export XDG_SESSION_TYPE=wayland
            export QT_QPA_PLATFORM=wayland
            break 2
        done
    done
    # 没找到 Wayland 会话则退回 xcb（本地调试场景）
    [ -n "${WAYLAND_DISPLAY:-}" ] || export QT_QPA_PLATFORM=xcb
else
    export QT_QPA_PLATFORM=wayland
fi

# 板上系统已有中文字体（wqy/noto）；conda fontconfig 默认读自己前缀的配置，
# 显式指回系统配置，配合 main.py 内 QFontDatabase 注册双保险
export FONTCONFIG_FILE="/etc/fonts/fonts.conf"

# 摄像头不写死: main.py 运行时扫描 /dev/video*（仍可用 CAMERA_INDEX 覆盖）
cd "$DIR/code"
# -u: 关闭 stdout 缓冲，日志实时落盘（便于 adb 排查）
# "$@": 透传命令行参数（demo 录制: --video x.mp4 --demo --record DIR）
exec python3 -u main.py "$@"
