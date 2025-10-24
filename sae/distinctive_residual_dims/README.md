# Distinctive Residual Stream Dimension Analysis: Critical Findings

## Dataset-Specific vs Shared Representations

This analysis identified residual stream dimensions (8,192-dim) that **distinguish** honest from deceptive examples within each dataset by ranking dimensions by activation difference (deceptive_mean - honest_mean). Unlike the SAE feature analysis which found highly task-specific representations, the residual stream shows substantially more cross-dataset overlap. Across 6 deception detection datasets, we found 224 unique dimensions that are distinctive for deceptive examples and similar numbers for honest examples. Critically, **40 dimensions (18%) appear as deceptive-distinctive in 2+ datasets**, with multiple dimensions appearing in 4-5 datasets. This is 2.5x higher overlap than SAE features (7%), suggesting the residual stream contains more general, reusable patterns that the SAE decomposes into task-specific sparse features.

## Why This Matters for Probe Generalization

The higher cross-dataset overlap in residual stream space (18% vs 7% for SAE features) helps explain why probes trained directly on residual stream activations show some (though limited) cross-task transfer. Dimensions like 6857, 3099, 4482, 6970, and 3222 appear as deceptive-distinctive in 5 out of 6 datasets, providing shared signal across multiple deception tasks. However, the most striking finding is **dimension 5114**, which shows opposite polarity across datasets: it's deceptive-distinctive for insider-trading, instructed-deception, and harm-pressure-knowledge-report, but honest-distinctive for convincing-game, harm-pressure-choice, and soft-trigger. This polarity flip suggests some residual stream dimensions encode general "deviation from baseline" that manifests as deception in some contexts and honesty in others, explaining why even residual-stream probes struggle to generalize perfectly.

## Comparison with SAE Feature Analysis

The SAE appears to be **decomposing** shared residual stream dimensions into more semantically specific sparse features. While residual dimension 6857 appears in 5 datasets, the corresponding SAE features are task-specific: insider-trading uses "professional rationalization" features, instructed-deception uses "verification and accuracy" features, and convincing-game uses "boundary-setting" features. This decomposition increases interpretability (SAE features have clear semantic labels) at the cost of transferability (features become task-specific). The residual stream dimensions show broader, more abstract patterns with signal strengths 10-100x larger than SAE features (Δ ≈ 7-76 for residual vs Δ ≈ 0.09-1.08 for SAE features), indicating the SAE is distributing a single strong residual stream signal across many weak sparse features.

## Residual Stream Dimensions by Dataset

Unlike SAE features which have semantic labels from Goodfire, residual stream dimensions are unlabeled. However, we can characterize them by their cross-dataset presence and signal strength:

### insider-trading
**Top deceptive-distinctive dimensions:**
- Dim 5114 (Δ=+7.88): Appears in 3 datasets as deceptive-distinctive
- Dim 7306 (Δ=+3.61): Appears in 4 datasets
- Dim 7033 (Δ=+2.67): Appears in 2 datasets

**Top honest-distinctive dimensions:**
- Dim 155 (Δ=-3.72): Appears in 2 datasets, but flips to deceptive in convincing-game
- Dim 4030 (Δ=-3.60): Dataset-specific
- Dim 6892 (Δ=-2.05): Dataset-specific

**Pattern**: Weak differences (Δ ≈ 2-8), suggesting subtle residual stream changes

### instructed-deception
**Top deceptive-distinctive dimensions:**
- Dim 5114 (Δ=+76.57): **Strongest signal across all datasets**, appears in 3 as deceptive
- Dim 6970 (Δ=+41.73): Appears in 5 datasets
- Dim 6857 (Δ=+34.11): Appears in 5 datasets

**Top honest-distinctive dimensions:**
- Dim 620 (Δ=-28.76): Appears in 2 datasets
- Dim 6853 (Δ=-28.61): Appears in 2 datasets
- Dim 4385 (Δ=-23.08): Appears in 3 datasets, flips to deceptive in convincing-game

**Pattern**: Very strong differences (Δ ≈ 23-76), clearest residual stream signal

### convincing-game
**Top deceptive-distinctive dimensions:**
- Dim 3467 (Δ=+79.82): **Strongest deceptive signal**, appears in 2 datasets
- Dim 155 (Δ=+39.48): **Polarity flip** from honest in insider-trading
- Dim 4385 (Δ=+32.82): **Polarity flip** from honest in other datasets

**Top honest-distinctive dimensions:**
- Dim 5114 (Δ=-186.84): **Strongest honest signal**, **polarity flip** from deceptive in 3 other datasets
- Dim 7033 (Δ=-44.55): Polarity flip
- Dim 6857 (Δ=-40.99): Polarity flip from deceptive in 5 other datasets

**Pattern**: Extremely strong differences (Δ ≈ 40-186), extensive polarity flips suggest convincing-game activates residual stream differently

### harm-pressure-choice
**Top deceptive-distinctive dimensions:**
- Dim 3467 (Δ=+9.12): Appears in 2 datasets
- Dim 6970 (Δ=+7.99): Appears in 5 datasets
- Dim 155 (Δ=+6.72): Mixed polarity across datasets

**Top honest-distinctive dimensions:**
- Dim 5114 (Δ=-38.51): Polarity flip
- Dim 4031 (Δ=-15.33): Appears in 3 datasets
- Dim 2567 (Δ=-5.74): Appears in 4 datasets

**Pattern**: Weak differences (Δ ≈ 6-38), small deceptive class (32 examples) may reduce signal

### harm-pressure-knowledge-report
**Top deceptive-distinctive dimensions:**
- Dim 5114 (Δ=+71.54): Strong deceptive signal, consistent with instructed-deception
- Dim 4994 (Δ=+33.17): Appears in 4 datasets
- Dim 6857 (Δ=+26.55): Appears in 5 datasets

**Top honest-distinctive dimensions:**
- Dim 4031 (Δ=-25.88): Appears in 3 datasets
- Dim 620 (Δ=-23.20): Appears in 2 datasets
- Dim 4385 (Δ=-20.50): Mixed polarity

**Pattern**: Strong differences (Δ ≈ 20-71), similar to instructed-deception

### soft-trigger
**Top deceptive-distinctive dimensions:**
- Dim 6857 (Δ=+22.76): Appears in 5 datasets
- Dim 6970 (Δ=+14.77): Appears in 5 datasets
- Dim 3099 (Δ=+11.91): Appears in 5 datasets

**Top honest-distinctive dimensions:**
- Dim 5114 (Δ=-34.86): Polarity flip
- Dim 4031 (Δ=-23.11): Appears in 3 datasets
- Dim 6853 (Δ=-15.49): Appears in 2 datasets

**Pattern**: Moderate differences (Δ ≈ 12-34), balanced classes (250/250)

## The Polarity Flip Phenomenon

**Dimension 5114** appears as:
- **Deceptive-distinctive**: insider-trading (+7.88), instructed-deception (+76.57), harm-pressure-knowledge-report (+71.54)
- **Honest-distinctive**: convincing-game (-186.84), harm-pressure-choice (-38.51), soft-trigger (-34.86)

This dimension shows the strongest signals in the entire analysis (ranging from -186 to +76), but with **opposite polarity** depending on the task. Similarly, dimensions 155 and 4385 flip polarity across datasets. This suggests these dimensions encode high-level features like "response elaboration," "compliance," or "deviation from typical response" that correlate with deception in some contexts and honesty in others. The polarity flip explains why linear probes trained on one dataset fail catastrophically on another (the same dimension has opposite meaning), and why even well-generalizing features don't guarantee transferable probes.

## Key Observations

1. **Higher cross-dataset reuse**: 18% of dimensions appear in multiple datasets (vs 7% for SAE features)
2. **Stronger signal magnitudes**: Residual stream differences are 10-100x larger than SAE feature differences
3. **Polarity flips**: Critical dimensions have opposite correlation with deception across tasks
4. **SAE decomposition**: The SAE appears to decompose shared residual dimensions into task-specific sparse patterns
5. **Interpretability tradeoff**: Residual stream has better transfer but no semantic labels; SAE features have labels but are task-specific

The residual stream analysis reveals why deception detection is fundamentally a context-dependent problem even at the raw activation level, before SAE decomposition.
