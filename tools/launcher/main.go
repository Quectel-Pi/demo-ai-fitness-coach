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
	"syscall"
)

func main() {
	self, err := os.Executable()
	if err != nil {
		fmt.Fprintln(os.Stderr, "[launcher] 找不到自身路径:", err)
		os.Exit(1)
	}
	root := filepath.Dir(filepath.Dir(self)) // <root>/code/ai-fitness-coach -> <root>

	// 单实例: NPU/摄像头不能被两个进程同时占用，双击=重启
	if err := exec.Command("pkill", "-f", "[m]ain.py").Run(); err == nil {
		fmt.Fprintln(os.Stderr, "[launcher] 已停止旧实例")
	}

	// 日志落盘（双击启动无终端，stdout 要有去处）
	logDir := filepath.Join(os.Getenv("HOME"), ".local", "state", "fitness-coach")
	_ = os.MkdirAll(logDir, 0o755)
	logFile, err := os.OpenFile(filepath.Join(logDir, "app.log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		fmt.Fprintln(os.Stderr, "[launcher] 打不开日志:", err)
		os.Exit(1)
	}
	defer logFile.Close()

	// 转发命令行参数（demo 录制: ai-fitness-coach --video x.mp4 --demo --record DIR）
	cmd := exec.Command(filepath.Join(root, "code", "run.sh"), os.Args[1:]...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.Stdin = nil
	// setsid: 完全脱离本次进程组，文件管理器/启动器退出后应用仍存活
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(logFile, "[launcher] 启动失败:", err)
		os.Exit(1)
	}
	fmt.Fprintln(logFile, "[launcher] started pid", cmd.Process.Pid)
}
