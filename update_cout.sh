#!/bin/bash
# Hourly: write latest experiment results to cout.md and push to GitHub
PROJ_DIR="/hd/liujx/microbiome_llm_project"
OUTPUT="$PROJ_DIR/cout.md"
RESULT_DIR="$PROJ_DIR/experiments/results"
LOG_DIR="/tmp/claude-1001/-hd-liujx/bc726d57-6d96-428a-ab3e-77fecefd1baf/tasks"

cd "$PROJ_DIR" || exit 1

# Build report
echo "# ProCyon v2 — Experiment Log ($(date '+%Y-%m-%d %H:%M'))" > "$OUTPUT"
echo "" >> "$OUTPUT"

# Check if experiment is running
if pgrep -f "procyon_v2_final.py" > /dev/null; then
    echo "## Status: **RUNNING**" >> "$OUTPUT"
else
    echo "## Status: IDLE" >> "$OUTPUT"
fi
echo "" >> "$OUTPUT"

# Latest result files
echo "## Result Files" >> "$OUTPUT"
for f in "$RESULT_DIR"/procyon_v2_*.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f")
        mtime=$(stat -c %y "$f" 2>/dev/null | cut -d. -f1)
        acc=$(python3 -c "import json; d=json.load(open('$f'));
            a=d.get('mean','');
            if not a and 'enc_nl' in d: a=d['enc_nl']['accuracy'];
            if not a and 'results' in d and 'Mean' in d.get('results',{}): a=d['results']['Mean']['accuracy_mean'];
            print(f'{float(a):.4f}' if a else 'N/A')" 2>/dev/null)
        echo "- **$name** ($mtime): ACC=$acc" >> "$OUTPUT"
    fi
done
echo "" >> "$OUTPUT"

# Latest experiment output (last 30 lines from running task)
TASK_FILE=$(ls -t "$LOG_DIR"/*.output 2>/dev/null | head -1)
if [ -n "$TASK_FILE" ]; then
    echo "## Latest Output ($(basename "$TASK_FILE"))" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    tail -30 "$TASK_FILE" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
fi
echo "" >> "$OUTPUT"

echo "*Auto-generated at $(date '+%Y-%m-%d %H:%M:%S')*" >> "$OUTPUT"

# Git commit & push (only if there are changes)
git add cout.md 2>/dev/null
if git diff --cached --quiet 2>/dev/null; then
    # No changes
    :
else
    git commit -m "Hourly update: $(date '+%Y-%m-%d %H:%M')" 2>/dev/null
    git push origin main 2>/dev/null
fi
