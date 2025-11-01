#!/bin/bash
# Run all 6 self-interrogation modes sequentially

echo "Running all 6 self-interrogation modes..."
echo "This will take ~1 hour total (10 minutes per mode)"
echo ""

modes=("direct" "direct_with_system" "rolebreak" "rolebreak_with_system" "rolebreak_v2" "rolebreak_v2_with_system")

for mode in "${modes[@]}"; do
    echo "========================================"
    echo "Starting mode: $mode"
    echo "========================================"
    python interrogate_test_pairs.py --mode "$mode"

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $mode"
    else
        echo "✗ Failed: $mode"
        exit 1
    fi
    echo ""
done

echo "========================================"
echo "All modes completed!"
echo "========================================"
