// 智能健身教练 双击启动器
// 交叉编译: GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o ai-fitness-coach .
// 产物为静态 ELF —— GNOME 文件管理器双击二进制直接运行（无「运行/显示」弹框），
// 定位方式: 通过 /proc/self/exe 找到自身路径，上两级即应用根目录。
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"
	"time"
)

func main() {
	self, err := os.Executable()
	if err != nil {
		fmt.Fprintln(os.Stderr, "[launcher] 找不到自身路径:", err)
		os.Exit(1)
	}
	root := filepath.Dir(filepath.Dir(self)) // <root>/code/ai-fitness-coach -> <root>

	// 单实例: 停掉旧实例再启动（重复双击 = 重启）
	if err := exec.Command("pkill", "-f", "[m]ain.py").Run(); err == nil {
		fmt.Fprintln(os.Stderr, "[launcher] 已停止旧实例")
		// 杀掉持有 QNN 会话的旧实例后，CDSP(NPU) 需要约半分钟复位；
		// 立刻起新实例会在 QNN 图加载阶段静默退出——必须等复位完再启动
		// （延时可用 FITNESS_RESTART_DELAY 秒数覆盖，默认 30）
		delay := 30
		if v := os.Getenv("FITNESS_RESTART_DELAY"); v != "" {
			if n, e := strconv.Atoi(v); e == nil && n >= 0 {
				delay = n
			}
		}
		fmt.Fprintf(os.Stderr, "[launcher] 等待 %d 秒让 NPU 复位...\n", delay)
		time.Sleep(time.Duration(delay) * time.Second)
	}

	// 日志落盘（双击启动无终端，stdout 要有去处）。
	// 权限降级链: 若 HOME 下日志目录打不开（如 root 实例先跑过一次，
	// 目录属主变 root），改用 /tmp 兜底，再不行丢弃输出——日志问题
	// 绝不能阻断应用启动。
	logDir := filepath.Join(os.Getenv("HOME"), ".local", "state", "fitness-coach")
	_ = os.MkdirAll(logDir, 0o755)
	logFile, err := os.OpenFile(filepath.Join(logDir, "app.log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		fmt.Fprintln(os.Stderr, "[launcher] HOME 日志不可写，降级 /tmp:", err)
		logFile, err = os.OpenFile(filepath.Join("/tmp",
			fmt.Sprintf("fitness-coach-%d.log", os.Getuid())),
			os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "[launcher] 日志均不可写，输出丢弃:", err)
		logFile = nil
	}
	if logFile != nil {
		defer logFile.Close()
	}

	// 转发命令行参数（demo 录制: ai-fitness-coach --video x.mp4 --demo --record DIR）
	cmd := exec.Command(filepath.Join(root, "code", "run.sh"), os.Args[1:]...)
	cmd.Stdout, cmd.Stderr = logFile, logFile // nil -> /dev/null（Go 语义）
	cmd.Stdin = nil
	// setsid: 完全脱离本次进程组，文件管理器/启动器退出后应用仍存活
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(logFile, "[launcher] 启动失败:", err)
		os.Exit(1)
	}
	fmt.Fprintln(logFile, "[launcher] started pid", cmd.Process.Pid)
}
