#!/bin/bash
# Wait for the current experiment to complete, then run bootstrap analysis

RESULTS_DIR="results/interrogation/insider-trading/meta-llama-llama-3.3-70b-instruct/no_system/direct"
RESULTS_FILE="$RESULTS_DIR/results.json"

echo "Waiting for experiment to complete..."
echo "Checking for: $RESULTS_FILE"
echo ""

# Wait until the results file exists and is not being written to
while true; do
    if [ -f "$RESULTS_FILE" ]; then
        # Check if file size is stable (not being written to)
        SIZE1=$(stat -c%s "$RESULTS_FILE" 2>/dev/null || echo 0)
        sleep 5
        SIZE2=$(stat -c%s "$RESULTS_FILE" 2>/dev/null || echo 0)

        if [ "$SIZE1" -eq "$SIZE2" ] && [ "$SIZE1" -gt 0 ]; then
            echo "✓ Results file found and stable"
            break
        else
            echo "File still being written... (size: $SIZE2 bytes)"
        fi
    else
        echo "Waiting for results file... ($(date '+%H:%M:%S'))"
        sleep 30
    fi
done

echo ""
echo "Running bootstrap analysis..."
echo "=============================="
python bootstrap_analysis.py \
    --results-dir "$RESULTS_DIR" \
    --n-iterations 10000 \
    --seed 42

echo ""
echo "✓ Bootstrap analysis complete!"
echo "Results saved to: $RESULTS_DIR/bootstrap_results.json"
