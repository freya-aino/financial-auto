#!/usr/bin/env bash
# Secret scanner (spec 60 §6.7): no external tool, only git + grep.
#
#   scripts/secret-scan.sh                 scan every tracked file (working tree) — `just secret-scan`
#   scripts/secret-scan.sh --staged        scan what is staged for commit (the pre-commit hook)
#   scripts/secret-scan.sh --history       scan every commit's added lines (slow; audits the past)
#   scripts/secret-scan.sh --install-hook  install .git/hooks/pre-commit running `--staged`
#
# Findings: credential-shaped strings (cloud / VCS / chat tokens, private keys, JWTs, Garage
# keys, quoted values assigned to *key / *secret / *token / *password names) and files that must
# never be committed (config/secrets.toml, .env*, private keys). A line containing
# `secret-scan: allow` is skipped (use it for documented placeholders). Exit 1 on any finding.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

# name|ERE — keep the expressions POSIX-ERE (git grep -E / grep -E).
PATTERNS=(
  'private key|-----BEGIN ([A-Z]+ )?PRIVATE KEY( BLOCK)?-----'
  'AWS access key id|(AKIA|ASIA)[0-9A-Z]{16}'
  'GitHub token|gh[pousr]_[A-Za-z0-9]{36,}'
  'GitHub fine-grained token|github_pat_[A-Za-z0-9_]{60,}'
  'GitLab token|glpat-[A-Za-z0-9_-]{20,}'
  'Slack token|xox[abprs]-[A-Za-z0-9-]{10,}'
  'Google API key|AIza[0-9A-Za-z_-]{35}'
  'Stripe live key|(sk|rk)_live_[0-9A-Za-z]{20,}'
  'OpenAI/Anthropic key|(^|[^A-Za-z0-9_-])sk-(ant-|proj-)?[A-Za-z0-9_-]*[0-9][A-Za-z0-9_-]*'
  'JWT|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}'
  'Garage key id|GK[0-9a-f]{24}'
)
# Quoted values (≥ 16 chars) assigned to credential-like names; matched case-insensitively.
# Plain `token` is not one of them: NATS subject tokens are 32-hex ids all over the fixtures.
ASSIGNED='(api[_-]?key|apikey|secret|secret[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|admin[_-]?token|bearer[_-]?token|refresh[_-]?token|private[_-]?key|password|passwd)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"'][A-Za-z0-9+/_=.-]{16,}["'"'"']'
# Matched lines that are placeholders, not secrets (case-insensitive).
PLACEHOLDER='(change-?me|example|placeholder|dummy|redacted|your[_-]|xxxx|\$\{|<[a-z_]+>|fake)'
# Paths that must never be committed (ERE on the path).
FORBIDDEN_PATHS='(^|/)(config/secrets\.toml|\.env(\.[^/]*)?|id_(rsa|ed25519|ecdsa)[^/]*|[^/]*\.(pem|p12|pfx|key))$'
FORBIDDEN_OK='(^|/)\.env\.example$'
ALLOW_MARK='secret-scan: allow'
SELF='scripts/secret-scan.sh'

findings=0
report() { # text, rule
  printf '%s  [%s]\n' "$1" "$2" >&2
  findings=$((findings + 1))
}

# Reads `<where>:<text>` lines that matched at least one pattern; names the rule, drops
# allow-marked lines, the scanner itself and placeholder values (e.g. AWS's documented
# AKIA…EXAMPLE key). Run with `< <(…)` so the counter survives (no pipeline subshell).
classify() {
  local line entry rule re
  while IFS= read -r line; do
    [[ $line == *"$ALLOW_MARK"* || $line == "$SELF:"* ]] && continue
    shopt -s nocasematch
    if [[ $line =~ $PLACEHOLDER ]]; then
      shopt -u nocasematch
      continue
    fi
    shopt -u nocasematch
    for entry in "${PATTERNS[@]}"; do
      rule=${entry%%|*} re=${entry#*|}
      if [[ $line =~ $re ]]; then
        report "${line:0:200}" "$rule"
        continue 2
      fi
    done
    shopt -s nocasematch
    if [[ $line =~ $ASSIGNED ]]; then
      report "${line:0:200}" "assigned secret"
    fi
    shopt -u nocasematch
  done
}

check_paths() { # stdin: paths
  local p
  while IFS= read -r p; do
    [[ -z $p ]] && continue
    if [[ $p =~ $FORBIDDEN_PATHS && ! $p =~ $FORBIDDEN_OK ]]; then
      report "$p" "file must not be committed"
    fi
  done
}

grep_args() { # every pattern as -e
  local entry
  for entry in "${PATTERNS[@]}"; do printf -- '-e\n%s\n' "${entry#*|}"; done
}

scan_tree() { # $@: extra git grep args (--cached for the index)
  local args=()
  mapfile -t args < <(grep_args)
  classify < <(
    git grep -nIE "$@" "${args[@]}" -- . ":(exclude)$SELF" 2>/dev/null
    git grep -niIE "$@" -e "$ASSIGNED" -- . ":(exclude)$SELF" 2>/dev/null
  )
}

case "${1:-}" in
  --install-hook)
    hook="$(git rev-parse --git-path hooks)/pre-commit"
    if [[ -e $hook ]] && ! grep -q 'secret-scan.sh' "$hook"; then
      echo "secret-scan: $hook exists and is not ours; add 'scripts/secret-scan.sh --staged || exit 1' to it" >&2
      exit 1
    fi
    cat >"$hook" <<'HOOK'
#!/bin/sh
# installed by scripts/secret-scan.sh --install-hook (spec 60 §6.7); bypass once: git commit --no-verify
exec "$(git rev-parse --show-toplevel)/scripts/secret-scan.sh" --staged
HOOK
    chmod +x "$hook"
    echo "secret-scan: pre-commit hook installed at $hook"
    exit 0
    ;;
  --staged)
    check_paths < <(git diff --cached --name-only --diff-filter=ACMR)
    scan_tree --cached
    ;;
  --history)
    check_paths < <(git log --all --name-only --diff-filter=A --format= | sort -u)
    combined=$( (printf '%s\n' "${PATTERNS[@]}" | sed 's/^[^|]*|//'; echo "$ASSIGNED") | paste -sd'|')
    classify < <(git log --all -p --no-color --format='commit %H' -- . ":(exclude)$SELF" 2>/dev/null |
      RE="$combined" awk '/^commit /{c=substr($2,1,12)} /^\+\+\+ /{f=substr($2,3)} /^\+[^+]/ && $0 ~ ENVIRON["RE"] {print c ":" f ":" substr($0,2)}')
    ;;
  "" | --tree)
    check_paths < <(git ls-files)
    scan_tree
    ;;
  *)
    sed -n '2,12p' "$0" >&2
    exit 2
    ;;
esac

if ((findings > 0)); then
  echo "secret-scan: $findings finding(s). Move secrets to config/secrets.toml or FA_* env (spec 60 §6.7);" >&2
  echo "             mark a documented placeholder with '$ALLOW_MARK'." >&2
  exit 1
fi
echo "secret-scan: clean"
