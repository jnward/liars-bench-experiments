#!/usr/bin/env python3
"""Compare Claude Sonnet 4.5, Gemma 2 27B, and Gemma 3 27B."""

import json
import numpy as np
from sklearn.metrics import roc_auc_score

print('='*80)
print('COMPREHENSIVE COMPARISON: Claude vs Gemma 2 vs Gemma 3')
print('='*80)
print()

models = [
    ('Claude Sonnet 4.5', 'results-claude-sonnet-4.5/test-pairs-organism-fda-Claude'),
    ('Gemma 2 27B', 'results-gemma-2-27b/test-pairs-organism-fda'),
    ('Gemma 3 27B', 'results-gemma-3-27b/test-pairs-organism-fda')
]

modes = [
    ('full_with_system', 'Full with System'),
    ('full_conversation', 'Full Conversation'),
    ('final_only', 'Final Only'),
    ('prompt_response_no_thinking', 'Prompt & Response (No Thinking)')
]

for mode_id, mode_name in modes:
    print(f'{mode_name}:')
    print('-' * 80)
    for model_name, base_path in models:
        path = f'{base_path}/score_based_v2_{mode_id}/results.json'
        try:
            with open(path) as f:
                results = json.load(f)

            valid = [r for r in results if r.get('prediction', -1) != -1]
            preds = np.array([r['prediction'] for r in valid])
            labels = np.array([r['true_label'] for r in valid])

            auroc = roc_auc_score(labels, preds)
            binary_preds = (preds >= 0.5).astype(int)
            accuracy = (binary_preds == labels).mean()

            print(f'  {model_name:<20} AUROC: {auroc:.4f} ({auroc*100:.1f}%)  |  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)')
        except Exception as e:
            print(f'  {model_name:<20} ERROR: {e}')
    print()

print('='*80)
print('\nKEY OBSERVATIONS:')
print('='*80)
print('1. Gemma 2 27B strongly outperforms both Claude and Gemma 3')
print('2. Gemma 3 27B shows unusual behavior: perfect AUROC but low accuracy')
print('   - This suggests Gemma 3 has perfect ranking but uses wrong threshold')
print('   - Or it may be predicting all/mostly one class')
print('3. Claude performs reasonably but is outperformed by Gemma 2')
print('='*80)
