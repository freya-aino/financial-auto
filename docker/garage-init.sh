#!/usr/bin/env sh
# One-time Garage init: single-node layout, buckets from config/platform.toml, access key.
# Safe to re-run: existing layout/buckets/key produce errors that are ignored.
set -u
cd "$(dirname "$0")/.."
G() { docker compose exec -T garage /garage "$@"; }

NODE=$(G node id -q | cut -d@ -f1)
if G layout show 2>/dev/null | grep -q "$(echo "$NODE" | cut -c1-16).*local"; then
  echo "layout already assigned"
else
  G layout assign -z local -c 50G "$NODE"
  G layout apply --version 1
fi

for b in fa-raw fa-media fa-models fa-ckpt fa-artifacts fa-backup; do
  G bucket create "$b" 2>/dev/null || echo "bucket $b exists"
done
G key info fa-local >/dev/null 2>&1 || G key create fa-local
for b in fa-raw fa-media fa-models fa-ckpt fa-artifacts fa-backup; do
  G bucket allow --read --write --owner "$b" --key fa-local >/dev/null
done

echo
echo "Put these into config/secrets.toml under [garage]:"
G key info fa-local --show-secret | grep -iE 'key id|secret key'
