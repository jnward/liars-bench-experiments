#!/bin/bash
# Run remaining 5 self-interrogation modes sequentially
# (direct mode already completed with AUROC: 0.9644)

set -e  # Exit on any error

echo "========================================"
echo "Running Remaining 5 Interrogation Modes"
echo "========================================"
echo "Estimated time: ~50 minutes total"
echo ""

modes=("direct_with_system" "rolebreak" "rolebreak_with_system" "rolebreak_v2" "rolebreak_v2_with_system")

for i in "${!modes[@]}"; do
    mode="${modes[$i]}"
    echo ""
    echo "========================================"
    echo "[$((i+1))/5] Starting mode: $mode"
    echo "========================================"

    python interrogate_test_pairs.py --mode "$mode"

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $mode"
    else
        echo "✗ Failed: $mode"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "All 5 modes completed!"
echo "========================================"
echo ""
echo "Summary of all 6 modes:"
echo "1. direct - COMPLETED (AUROC: 0.9644)"
echo "2. direct_with_system - COMPLETED"
echo "3. rolebreak - COMPLETED"
echo "4. rolebreak_with_system - COMPLETED"
echo "5. rolebreak_v2 - COMPLETED"
echo "6. rolebreak_v2_with_system - COMPLETED"
