#!/usr/bin/env python3
"""解析 conda-forge pyside6 (linux-aarch64, 目标板上 Python 3.13) 的依赖闭包，输出下载清单。

不安装任何东西到宿主机，只输出 文件名\t大小 列表。
用法: python3 tools/resolve_conda_deps.py > pkgs.txt

repodata 缓存（cf-repo.json）查找顺序:
  1. 环境变量 FITNESS_REPODATA 指定的路径
  2. /home/q/work/temp/tmp/fitness-coach/cf-repo.json（存在时用，本机惯例位置）
  3. <项目>/.cache/cf-repo.json —— 不存在时自动从清华镜像下载（一次性，约 30MB）
"""
import json
import os
import re
import sys
import urllib.request

REPODATA_MIRROR = ("https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/"
                   "conda-forge/linux-aarch64/repodata.json")


def _load_repodata():
    cands = []
    if os.environ.get("FITNESS_REPODATA"):
        cands.append(os.environ["FITNESS_REPODATA"])
    cands.append("/home/q/work/temp/tmp/fitness-coach/cf-repo.json")
    cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", ".cache", "cf-repo.json"))
    for path in cands:
        if os.path.exists(path):
            print(f"[deps] 使用 repodata 缓存: {path}", file=sys.stderr)
            return json.load(open(path))
    # 缓存全无 -> 自动下载到最后一个候选（项目内 .cache/，可重复利用）
    path = cands[-1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[deps] 无 repodata 缓存，从镜像下载（一次性，约 30MB）...", file=sys.stderr)
    urllib.request.urlretrieve(REPODATA_MIRROR, path)
    print(f"[deps] 已缓存到: {path}", file=sys.stderr)
    return json.load(open(path))


REPO = _load_repodata()
ALL = {**REPO.get("packages", {}), **REPO.get("packages.conda", {})}

BY_NAME = {}
for fname, info in ALL.items():
    info["fn"] = fname  # repodata 的键就是文件名
    BY_NAME.setdefault(info["name"], []).append(info)

SKIP = {"python", "python_abi", "__glibc", "__unix", "__linux", "__linux-64",
        "__linux-aarch64", "sysroot_linux-64", "fonts-conda-ecosystem"}


def vkey(v):
    parts = []
    for x in re.split(r"[._]", v):
        parts.append(int(x) if x.isdigit() else 0)
    return tuple(parts)


def parse_dep(dep):
    parts = dep.split()
    if not parts:
        return dep, []
    name = parts[0]
    cons = []
    i = 1
    while i < len(parts):
        consumed_build = False
        # conda 约束可能逗号连接在同一 token 里（如 "icu >=78.3,<79.0a0"），
        # 必须逐段解析——否则复合约束整体匹配失败被丢弃，版本 pin 全部失效
        for sub in parts[i].split(","):
            sub = sub.strip()
            if not sub:
                continue
            m = re.match(r"^([=<>!~]+)([0-9a-zA-Z.*+_-]+)$", sub)
            if m:
                cons.append((m.group(1), m.group(2)))
            elif re.match(r"^[0-9]", sub):
                cons.append(("==", sub))
                # 版本串后跟构建串 = 精确钉住 (name 1.2.3 h123_0)——仅无逗号场景
                if ("," not in parts[i] and i + 1 < len(parts)
                        and re.match(r"^[a-z0-9_]+$", parts[i + 1])
                        and not parts[i + 1][0].isdigit()):
                    cons.append(("build", parts[i + 1]))
                    consumed_build = True
        i += 2 if consumed_build else 1
    return name, cons


def satisfies(pkg, cons):
    for op, val in cons:
        if op == "build":
            if pkg.get("build") != val:
                return False
            continue
        try:
            if op == "==":
                if not pkg["version"].startswith(val.rstrip(".*")):
                    return False
            elif op in (">=", ">", "<=", "<"):
                a, b = vkey(pkg["version"]), vkey(val.rstrip(".*"))
                if op == ">=" and not a >= b:
                    return False
                if op == ">" and not a > b:
                    return False
                if op == "<=" and not a <= b:
                    return False
                if op == "<" and not a < b:
                    return False
            elif op == "!=":
                if pkg["version"].startswith(val.rstrip(".*")):
                    return False
        except Exception:
            pass
    return True


def is_python_pkg(pkg):
    deps = pkg.get("depends", [])
    return any(d.split()[0] in ("python", "python_abi") for d in deps if d)


def pick(name, cons, skip_fns=frozenset()):
    cands = [p for p in BY_NAME.get(name, [])
             if p["fn"] not in skip_fns and satisfies(p, cons)]
    if not cands:
        return None
    cands.sort(key=lambda p: (vkey(p["version"]), p.get("build_number", 0)), reverse=True)
    # python 绑定包：必须选 py313 构建（板上 python 3.13）
    if is_python_pkg(cands[0]):
        py313 = [p for p in cands if "py313" in p.get("build", "")]
        if py313:
            return py313[0]
        # 没标 py313 的 python 包（纯 py 代码等）直接用最新
        return cands[0]
    return cands[0]


resolved = {}
cons_map = {}      # name -> 累积约束列表（多个包依赖同一库时约束必须合并，
                   #   否则 DFS 先到先得会丢弃后来者的版本 pin——如 qt6-main
                   #   要求 icu >=78.3,<79，若先被宽松约束解析成 79.1 就会
                   #   导致 PySide6 运行时缺 libicui18n.so.78 直接 ImportError）
excluded = {}     # name -> 与已选依赖库冲突、被降级排除的版本集合
dependents = {}   # 库名 -> 依赖它的包名列表（库重选时反向唤醒重新检查）


def solve(root):
    from collections import deque
    queue = deque([(root, [])])
    guard = 20000        # 防 A 降级 B、B 降级 A 的振荡死循环（正常远用不到）
    while queue and guard > 0:
        guard -= 1
        name, cons = queue.popleft()
        if name in SKIP:
            continue
        merged = cons_map.setdefault(name, [])
        for c in cons:
            if c not in merged:
                merged.append(c)
        # 带一致性检查的选包：候选包依赖的库若已全局选定，版本必须兼容；
        # 不兼容则按 fn 排除该候选（同版本不同 build 对应不同依赖 ABI，
        # 排除粒度必须到 fn 才能选到旧 build），自动降级——模拟 conda 求解器回退
        skip = set(excluded.get(name, ()))
        while True:
            pkg = pick(name, merged, skip)
            if pkg is None:
                print(f"WARNING: missing {name} {merged}", file=sys.stderr)
                break
            bad_dep = None
            for dep in pkg.get("depends", []):
                dname, dcons = parse_dep(dep)
                if dname in resolved and not satisfies(resolved[dname], dcons):
                    bad_dep = dname
                    break
            if bad_dep is None:
                break
            skip.add(pkg["fn"])
        if pkg is None:
            continue
        excluded[name] = skip
        old = resolved.get(name)
        if old is None or old["fn"] != pkg["fn"]:
            resolved[name] = pkg
            for dep in pkg.get("depends", []):
                dname, dcons = parse_dep(dep)
                dependents.setdefault(dname, []).append(name)
                queue.append((dname, dcons))
            # 本包定版/换版 -> 已选的其他包若依赖本包且约束不再满足，重新入队检查
            for other_name, other_pkg in resolved.items():
                if other_name == name:
                    continue
                for dep in other_pkg.get("depends", []):
                    dname, dcons = parse_dep(dep)
                    if dname == name and not satisfies(pkg, dcons):
                        queue.append((other_name, list(cons_map.get(other_name, []))))
                        break


solve("pyside6")
for name, pkg in sorted(resolved.items()):
    print(f"{pkg['fn']}\t{pkg['size']}")
print(f"# total {sum(p['size'] for p in resolved.values()) // 1048576} MB, {len(resolved)} packages", file=sys.stderr)
