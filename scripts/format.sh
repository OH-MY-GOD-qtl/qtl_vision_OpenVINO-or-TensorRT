#!/usr/bin/env bash
# 按仓库根目录 .clang-format 校验 / 格式化 C/C++ 源码。
#
# 用法：
#   scripts/format.sh        只检查改动（相对 HEAD 的修改 + 未跟踪的新文件）
#   scripts/format.sh --all  检查全部受跟踪的 C/C++ 文件（跳过 third_party/）
#   scripts/format.sh --fix  就地格式化（默认只作用于改动文件，可与 --all 组合）
#
# 为什么要这个脚本：本机没有系统级 clang-format，但 C/C++ 扩展自带一份
# （~/.vscode-server/extensions/ms-vscode.cpptools-*/LLVM/bin/clang-format）。
# 编辑器一旦拿不到这份二进制，就可能退回到内置默认风格（K&R / 2 空格 / 短 if
# 合行），把整片代码重排成和 .clang-format 相反的样子。这里固定用 --style=file，
# 保证结果只由仓库里的 .clang-format 决定。

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 1

fix=0
all=0
for arg in "$@"; do
    case "$arg" in
        --fix) fix=1 ;;
        --all) all=1 ;;
        -h | --help) sed -n '2,13p' "$0"; exit 0 ;;
        *)
            echo "未知参数: $arg" >&2
            exit 2
            ;;
    esac
done

find_clang_format()
{
    if [[ -n ${CLANG_FORMAT:-} && -x ${CLANG_FORMAT} ]]; then
        echo "$CLANG_FORMAT"
        return 0
    fi

    local found
    if found=$(command -v clang-format); then
        echo "$found"
        return 0
    fi

    # 取最新安装的 cpptools 自带的 clang-format
    found=$(ls -1dt \
        "$HOME"/.vscode-server/extensions/ms-vscode.cpptools-*/LLVM/bin/clang-format \
        "$HOME"/.vscode/extensions/ms-vscode.cpptools-*/LLVM/bin/clang-format \
        2>/dev/null | head -1)
    [[ -n $found && -x $found ]] || return 1
    echo "$found"
}

CF=$(find_clang_format) || {
    echo "找不到 clang-format：请安装，或用 CLANG_FORMAT=/path/to/clang-format 指定" >&2
    exit 2
}

mapfile -t files < <(
    if [[ $all -eq 1 ]]; then
        git ls-files -- '*.cpp' '*.hpp' '*.h' '*.cc'
    else
        {
            git diff --name-only --diff-filter=ACMR HEAD -- '*.cpp' '*.hpp' '*.h' '*.cc'
            git ls-files --others --exclude-standard -- '*.cpp' '*.hpp' '*.h' '*.cc'
        } | sort -u
    fi | grep -vE '^(third_party|build)/' | { grep -E '\.(cpp|hpp|h|cc)$' || true; }
)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "没有需要检查的 C/C++ 文件"
    exit 0
fi

echo "clang-format: $CF ($("$CF" --version))"
echo "待处理文件: ${#files[@]}"

if [[ $fix -eq 1 ]]; then
    "$CF" -i --style=file "${files[@]}" || exit 1
    echo "已就地格式化"
    exit 0
fi

bad=()
for f in "${files[@]}"; do
    if ! "$CF" --dry-run --Werror --style=file "$f" >/dev/null 2>&1; then
        bad+=("$f")
    fi
done

if [[ ${#bad[@]} -eq 0 ]]; then
    echo "格式检查通过"
    exit 0
fi

echo "以下文件不符合 .clang-format：" >&2
printf '  %s\n' "${bad[@]}" >&2
echo "修复：$CF -i --style=file <文件>  或  scripts/format.sh --fix" >&2
exit 1
