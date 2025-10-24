# Distinctive Feature Analysis: Critical Findings

## Dataset-Specific Deception Representations

This analysis identified SAE features that **distinguish** honest from deceptive examples within each dataset by ranking features by activation difference (deceptive_mean - honest_mean) rather than raw activation magnitude. Across 6 deception detection datasets, we found 279 unique features that are distinctive for deceptive examples and 280 distinctive for honest examples. However, the critical finding is that **only 20 features (7%) appear as deceptive-distinctive in 2+ datasets**, with just one feature (34806: "Offensive request from the user") appearing in 3 datasets. This means 93% of deceptive-distinctive features are dataset-specific, revealing that the model's internal representation of deception is highly context-dependent.

## Why Probes Don't Generalize

This feature sparsity explains the probe generalization failure observed in earlier experiments: single-dataset probes achieve 0.97 AUROC, while leave-one-out and combined probes drop to 0.50 AUROC. Each deception task (insider trading, instructed lying, convincing games) activates fundamentally different SAE features. For example, instructed-deception activates features related to "verifying factual claims" and "providing intentionally incorrect information," while convincing-game activates "establishing boundaries" and "rejecting inappropriate requests," and insider-trading activates "minimizing problematic behavior" and "high-stakes financial deadlines." These are semantically distinct representations that don't overlap at the feature level.

## Implications for Interpretability

This finding challenges the assumption that deception has a consistent neural representation across contexts. The semantic ablation experiments showed <1% impact when removing labeled feature categories (Evasion, Refusal, Meta, etc.), further confirming that probes rely on distributed, task-specific feature patterns rather than discrete semantic concepts. The top-N ablation by cosine similarity found critical features between ranks 1000-5000, but these features are dataset-specific, not general deception detectors. For mechanistic interpretability of deception detection, this suggests we need task-specific analysis rather than universal deception features, and that SAE features, while interpretable individually, compose into task-specific patterns that don't transfer across deception contexts.
