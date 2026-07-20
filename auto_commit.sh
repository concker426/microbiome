#!/bin/bash
# Every 30 min: commit all changes and push to GitHub
cd /hd/liujx/microbiome_llm_project || exit 1
git add -A 2>/dev/null
if git diff --cached --quiet 2>/dev/null; then
    exit 0  # no changes
fi
git commit -m "Auto: $(date '+%Y-%m-%d %H:%M')" 2>/dev/null
git push origin main 2>/dev/null
