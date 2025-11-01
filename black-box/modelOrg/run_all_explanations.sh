#!/bin/bash
# Get explanations from Claude judge for all 4 context modes IN PARALLEL

echo "========================================"
echo "Getting Claude Judge Explanations"
echo "========================================"
echo "Running all 4 modes IN PARALLEL"
echo "Estimated time: ~30 seconds total"
echo ""

# Launch all 4 modes in parallel
echo "Launching parallel jobs..."
python explain_judge_reasoning.py --mode full_with_system > logs/explain_full_with_system.log 2>&1 &
pid1=$!
python explain_judge_reasoning.py --mode full_conversation > logs/explain_full_conversation.log 2>&1 &
pid2=$!
python explain_judge_reasoning.py --mode final_only > logs/explain_final_only.log 2>&1 &
pid3=$!
python explain_judge_reasoning.py --mode prompt_response_no_thinking > logs/explain_prompt_response_no_thinking.log 2>&1 &
pid4=$!

echo "✓ Launched 4 parallel processes"
echo "  - full_with_system (PID: $pid1)"
echo "  - full_conversation (PID: $pid2)"
echo "  - final_only (PID: $pid3)"
echo "  - prompt_response_no_thinking (PID: $pid4)"
echo ""
echo "Waiting for all to complete..."

# Wait for all background jobs
wait $pid1
status1=$?
wait $pid2
status2=$?
wait $pid3
status3=$?
wait $pid4
status4=$?

echo ""
echo "========================================"
echo "Results:"
echo "========================================"
[ $status1 -eq 0 ] && echo "✓ full_with_system" || echo "✗ full_with_system (failed)"
[ $status2 -eq 0 ] && echo "✓ full_conversation" || echo "✗ full_conversation (failed)"
[ $status3 -eq 0 ] && echo "✓ final_only" || echo "✗ final_only (failed)"
[ $status4 -eq 0 ] && echo "✓ prompt_response_no_thinking" || echo "✗ prompt_response_no_thinking (failed)"

echo ""
echo "Logs saved to:"
echo "  logs/explain_*.log"
echo ""
echo "Results saved to:"
echo "  results-claude-sonnet-4.5/test-pairs-organism-fda/score_based_v2_*/results_with_explanations.json"

# Exit with error if any job failed
if [ $status1 -ne 0 ] || [ $status2 -ne 0 ] || [ $status3 -ne 0 ] || [ $status4 -ne 0 ]; then
    exit 1
fi
