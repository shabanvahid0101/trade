#!/usr/bin/env bash
# Called under the repository's shared writer concurrency group.
set -euo pipefail
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
if [ "${1:-snapshot}" = "receipts" ]; then
  paths=(reports/bittrade/paper_state.json)
  message="Record BitTrade Telegram delivery receipts"
else
  paths=(dataset/bittrade reports/bittrade/research_report.json reports/bittrade/SUMMARY.md
         reports/bittrade/paper_state.json reports/bittrade/PAPER_SUMMARY.md)
  message="Refresh BitTrade research and forward paper ledger"
fi
for path in "${paths[@]}"; do
  if [ -e "$path" ]; then
    git add -- "$path"
  fi
done
if git diff --cached --quiet; then
  exit 0
fi
git commit -m "$message"
for attempt in 1 2 3; do
  git pull --rebase origin main
  if git push origin HEAD:main; then
    exit 0
  fi
  sleep 5
done
exit 1
