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

# ---- 大小核绑定 ----
# H1(QCS6490) 是 4x A55 小核 + 4x A78 大核；ORT CPU 检测头线程默认会被
# 调度器放到小核（板子发热后更明显），推理从 ~36ms 掉到 ~160ms。
# 自动挑主频最高的一组核绑上去（绑不上则原样运行，不影响功能）。
BIG_CORES=""
if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    MAX_F=0
    for c in /sys/devices/system/cpu/cpu[0-9]*; do
        f=$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null) || continue
        [ -n "$f" ] && [ "$f" -gt "$MAX_F" ] && MAX_F=$f
    done
    # 大核簇 = 主频 >= 0.8x 最高主频的核（H1: A78 2.4G/2.7G 均入选，A55 1.96G 排除）
    if [ "$MAX_F" -gt 0 ]; then
        TH=$(( MAX_F * 8 / 10 ))
        for c in /sys/devices/system/cpu/cpu[0-9]*; do
            f=$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null) || continue
            [ -n "$f" ] && [ "$f" -ge "$TH" ] && BIG_CORES="${BIG_CORES:+$BIG_CORES,}${c##*cpu}"
        done
    fi
fi
cd "$DIR/code"
# ---- 清理旧实例 ----
# 摄像头是独占资源：旧实例残留会导致新实例打不开摄像头（画面卡住）。
# 用 SIGTERM 优雅退出并等待；严禁 SIGKILL（强杀持有 QNN 会话的进程会
# 把 CDSP 打成脏态，之后 NPU 初始化 RPC 永不返回）。[m] 写法避免自匹配。
OLD_PIDS=$(pgrep -f 'python3 -u [m]ain.py' || true)
if [ -n "$OLD_PIDS" ]; then
    echo "[run] stopping old instance: $OLD_PIDS"
    kill $OLD_PIDS 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        pgrep -f 'python3 -u [m]ain.py' >/dev/null || break
        sleep 0.5
    done
    # 仍未退出再补一次 TERM（依旧不用 KILL，保护 CDSP）
    pkill -f 'python3 -u [m]ain.py' 2>/dev/null || true
    sleep 1
    # 杀掉持有 QNN 会话的旧实例后，CDSP(NPU) 需要约半分钟复位，
    # 立刻起新实例会在 QNN 图加载阶段静默退出（与 launcher 的延时逻辑同源，
    # 可用 FITNESS_RESTART_DELAY 秒数覆盖，默认 30）
    DELAY="${FITNESS_RESTART_DELAY:-30}"
    echo "[run] waiting ${DELAY}s for NPU reset"
    sleep "$DELAY"
    # 仍活着说明无权杀（如 q 用户遇到 root 实例）——打警告不阻断，
    # main.py 内置的摄像头重试会再兜底 15s
    if pgrep -f 'python3 -u [m]ain.py' >/dev/null; then
        echo "[run] WARN: old instance still alive (no permission?), camera may be busy"
    fi
fi
# -u: 关闭 stdout 缓冲，日志实时落盘（便于 adb 排查）
# "$@": 透传命令行参数（demo 录制: --video x.mp4 --demo --record DIR）
if [ -n "$BIG_CORES" ] && command -v taskset >/dev/null 2>&1; then
    echo "[run] bind to big cores: $BIG_CORES"
    exec taskset -c "$BIG_CORES" python3 -u main.py "$@"
fi
exec python3 -u main.py "$@"
