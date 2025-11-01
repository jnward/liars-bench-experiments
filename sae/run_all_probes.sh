#!/bin/bash

# Run similarity analysis for all probes
echo "Running SAE similarity analysis for all probes..."
echo "=================================================="

for probe in probes-layer50/*.pkl; do
    probe_name=$(basename "$probe")
    echo ""
    echo "Processing: $probe_name"
    echo "----------------------------------------"
    python compute_sae_similarities.py --probe "$probe"
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $probe_name"
    else
        echo "✗ Failed: $probe_name"
    fi
done

echo ""
echo "=================================================="
echo "All probes processed!"
echo "Results saved in outputs/ directory"
