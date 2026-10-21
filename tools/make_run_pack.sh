#!/bin/bash
# 生成「智能健身教练」自解压一键安装包 (.run)
# 产物: dist/ai-fitness-coach-installer.run
# 用法: tools/make_run_pack.sh
#   板上运行安装包: ./ai-fitness-coach-installer.run [目标目录]  默认 ~/ai-fitness-coach
#   root（装到 /root/ai-fitness-coach）与普通用户（装到 ~/ai-fitness-coach）均可
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/dist"
TMP="${FIT_PACK_TMP:-/home/q/work/temp/tmp/fitness-runpack}"   # 临时产物目录（可用环境变量覆盖）
NAME="ai-fitness-coach-installer.run"

# 不 rm -rf 临时目录（安全策略拦截），固定文件名覆盖写即可
mkdir -p "$TMP" "$DIST" 2>/dev/null || TMP="$(mktemp -d)"    # 默认路径不可写时回退系统临时目录
mkdir -p "$DIST"

# 1) payload: tar.xz（xz 多线程加速，~1GB runtime 压到 200MB 级）
echo "==> 打包 payload (tar.xz, 多线程)..."
XZ_OPT="-T0 -6" tar -cJf "$TMP/payload.tar.xz" -C "$ROOT" \
    --exclude='code/__pycache__' \
    code models audio assets runtime README.md

# 2) 安装器 header（自解压逻辑）
cat > "$TMP/header.sh" <<'HEADER'
#!/bin/bash
# 智能健身教练 一键安装器（自解压 .run）
# 用法: ./ai-fitness-coach-installer.run [目标目录]   默认 ~/ai-fitness-coach
# root 与普通用户均可执行
set -eu
TARGET="${1:-${HOME:-/root}/ai-fitness-coach}"
PAYLOAD_LINE=__SKIP__

# 防呆: 拒绝装到 / 或家目录本身
if [ "$TARGET" = "/" ] || [ "$TARGET" = "${HOME:-}" ]; then
    echo "!! 拒绝安装到 $TARGET，请指定子目录（默认 ~/ai-fitness-coach）"
    exit 1
fi

echo "==> 智能健身教练 安装到: $TARGET"
# 旧版本清理（仅当目录确实是本应用时才删除）
if [ -f "$TARGET/code/main.py" ]; then
    echo "==> 检测到旧版本，先清理..."
    rm -rf "$TARGET"
fi
mkdir -p "$TARGET"

echo "==> 解压中（约 1GB，需一两分钟）..."
# 板上最小系统无 xz 命令 -> python3 自带 lzma 流式解压
# 注意: 脚本必须用 -c 传参，stdin 留给管道里的 payload
tail -n "+$PAYLOAD_LINE" "$0" | DEST="$TARGET" python3 -c '
import os, sys, tarfile
dest = os.environ["DEST"]
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|xz") as t:
    try:
        t.extractall(dest, filter="tar")
    except TypeError:            # py<3.12 无 filter 参数
        t.extractall(dest)
'
chmod +x "$TARGET/code/run.sh" "$TARGET/code/ai-fitness-coach"

# root 装到 /home 下时把文件还给目录属主（否则 __pycache__/日志写不进去）
if [ "$(id -u)" = 0 ]; then
    case "$TARGET" in
        /home/*) chown -R "$(stat -c %u:%g "$(dirname "$TARGET")")" "$TARGET" 2>/dev/null || true ;;
    esac
fi

# 权限友好检查: video/audio 组（缺失时提示，不中断）
for g in video audio; do
    if ! id -nG | tr ' ' '\n' | grep -qx "$g"; then
        what="摄像头"; [ "$g" = audio ] && what="声卡"
        echo "!! 提示: 当前用户不在 $g 组，$what 可能不可用"
        echo "   修复: sudo usermod -aG $g $(id -un) 后重新登录"
    fi
done

echo ""
echo "==> 安装完成！进入 $TARGET 双击 code/ai-fitness-coach 即可运行"
echo "    终端启动: $TARGET/code/run.sh"
echo "    运行日志: ~/.local/state/fitness-coach/app.log"
exit 0
HEADER

# 3) header 行数 + 1 = payload 起始行（替换占位不改变行数）
SKIP=$(( $(wc -l < "$TMP/header.sh") + 1 ))
sed -i "s/__SKIP__/$SKIP/" "$TMP/header.sh"

# 4) 拼接: header + payload
cat "$TMP/header.sh" "$TMP/payload.tar.xz" > "$DIST/$NAME"
chmod +x "$DIST/$NAME"
echo "==> 生成完成: $DIST/$NAME ($(du -h "$DIST/$NAME" | cut -f1))"
