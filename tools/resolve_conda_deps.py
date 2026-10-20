#!/usr/bin/env python3
"""解析 conda-forge pyside6 (linux-aarch64, 目标板上 Python 3.13) 的依赖闭包，输出下载清单。

不安装任何东西到宿主机，只输出 文件名\t大小 列表。
用法: python3 tools/resolve_conda_deps.py > /tmp/pkgs.txt
"""
import json
import re
import sys

REPO = json.load(open("/home/q/work/temp/tmp/fitness-coach/cf-repo.json"))
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
        tok = parts[i]
        m = re.match(r"^([=<>!~]+)([0-9a-zA-Z.*+_-]+)$", tok)
        if m:
            cons.append((m.group(1), m.group(2)))
        elif re.match(r"^[0-9]", tok):
            cons.append(("==", tok))
            # 版本后跟构建串 = 精确钉住 (name 1.2.3 h123_0)
            if i + 1 < len(parts) and re.match(r"^[a-z0-9_]+$", parts[i + 1]) and not parts[i + 1][0].isdigit():
                cons.append(("build", parts[i + 1]))
                i += 1
        i += 1
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


def pick(name, cons):
    cands = [p for p in BY_NAME.get(name, []) if satisfies(p, cons)]
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


def solve(name, cons):
    if name in SKIP or name in resolved:
        return
    pkg = pick(name, cons)
    if pkg is None:
        print(f"WARNING: missing {name} {cons}", file=sys.stderr)
        return
    resolved[name] = pkg
    for dep in pkg.get("depends", []):
        dname, dcons = parse_dep(dep)
        if dname in SKIP:
            continue
        solve(dname, dcons)


solve("pyside6", [])
for name, pkg in sorted(resolved.items()):
    print(f"{pkg['fn']}\t{pkg['size']}")
print(f"# total {sum(p['size'] for p in resolved.values()) // 1048576} MB, {len(resolved)} packages", file=sys.stderr)
