#!/usr/bin/env sh
# Contract check (spec 21 §6, §8): `buf lint` + `buf breaking` of proto/ against a git ref
# (default main). This is the project's CI for contracts — there is no hosted CI; run it before
# committing a .proto change (`just proto-check`). Exit != 0 on any lint error or breaking change.
#
#   scripts/buf-check.sh                 # working tree vs main
#   scripts/buf-check.sh --against HEAD~3
#
# Uses a local `buf` when installed, else the bufbuild/buf image (FA_BUF_IMAGE). The comparison
# input is the repo's own .git (`.git#ref=<ref>,subdir=proto`), so uncommitted edits are checked
# against the committed contracts.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
against=main
while [ $# -gt 0 ]; do
    case "$1" in
        --against) against=$2; shift 2 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "unknown argument $1" >&2; exit 64 ;;
    esac
done
image=${FA_BUF_IMAGE:-bufbuild/buf:1.50.0}
input=".git#ref=$against,subdir=proto"
if command -v buf >/dev/null 2>&1; then
    run() { (cd "$root" && buf "$@"); }
else
    run() { docker run --rm -e HOME=/tmp -v "$root":/repo -w /repo "$image" "$@"; }
fi
echo "buf lint proto/"
run lint proto
echo "buf breaking proto/ against $against"
run breaking proto --against "$input"
echo "contracts OK (lint clean, no breaking change against $against)"
