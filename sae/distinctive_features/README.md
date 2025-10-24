# Distinctive Feature Analysis: Critical Findings

## Dataset-Specific Deception Representations

This analysis identified SAE features that **distinguish** honest from deceptive examples within each dataset by ranking features by activation difference (deceptive_mean - honest_mean) rather than raw activation magnitude. Across 6 deception detection datasets, we found 279 unique features that are distinctive for deceptive examples and 280 distinctive for honest examples. However, the critical finding is that **only 20 features (7%) appear as deceptive-distinctive in 2+ datasets**, with just one feature (34806: "Offensive request from the user") appearing in 3 datasets. This means 93% of deceptive-distinctive features are dataset-specific, revealing that the model's internal representation of deception is highly context-dependent.

## Why Probes Don't Generalize

This feature sparsity explains the probe generalization failure observed in earlier experiments: single-dataset probes achieve 0.97 AUROC, while leave-one-out and combined probes drop to 0.50 AUROC. Each deception task (insider trading, instructed lying, convincing games) activates fundamentally different SAE features. For example, instructed-deception activates features related to "verifying factual claims" and "providing intentionally incorrect information," while convincing-game activates "establishing boundaries" and "rejecting inappropriate requests," and insider-trading activates "minimizing problematic behavior" and "high-stakes financial deadlines." These are semantically distinct representations that don't overlap at the feature level.

## Implications for Interpretability

This finding challenges the assumption that deception has a consistent neural representation across contexts. The semantic ablation experiments showed <1% impact when removing labeled feature categories (Evasion, Refusal, Meta, etc.), further confirming that probes rely on distributed, task-specific feature patterns rather than discrete semantic concepts. The top-N ablation by cosine similarity found critical features between ranks 1000-5000, but these features are dataset-specific, not general deception detectors. For mechanistic interpretability of deception detection, this suggests we need task-specific analysis rather than universal deception features, and that SAE features, while interpretable individually, compose into task-specific patterns that don't transfer across deception contexts.

## Feature Categories by Dataset

Analysis of the top-10 deceptive-distinctive features for each dataset reveals distinct semantic patterns that explain why deception representations don't transfer across tasks:

### insider-trading
**Theme: Professional rationalization and structured justification**
- High-stakes work/financial context (deadlines, projects)
- Minimizing and excuse-making rhetoric
- Structured, templated responses with clear sections
- Task instruction framing and derivation phrases
- Subtle differences (Δ ≈ 0.07-0.09), suggesting nuanced deception

### instructed-deception
**Theme: Verification, accuracy manipulation, and explicit falsification**
- Requests for intentionally incorrect information (strongest signal: Δ=0.68)
- Verification of factual claims and consistency checking
- System metadata and knowledge boundaries
- Formal communication and structured organization
- Moderate-to-strong differences (Δ ≈ 0.45-0.73), clear deceptive intent

### convincing-game
**Theme: Boundary-setting, refusal patterns, and AI limitations**
- Offensive request detection (strongest signal: Δ=1.08)
- Statements about AI limitations and fallibility
- Rejecting harmful/illegal requests and establishing boundaries
- Coercive language and non-consensual action descriptions
- Strong differences (Δ ≈ 0.48-1.08), explicit adversarial pressure

### harm-pressure-choice
**Theme: Corrections, technical formatting, and conversation structure**
- Acknowledging mistakes and providing corrections
- Code completion and technical documentation patterns
- System configuration and conversation metadata
- Table formatting and technical identifiers
- Weak differences (Δ ≈ 0.16-0.24), ambiguous deception signals

### harm-pressure-knowledge-report
**Theme: Access control, compliance, and content moderation**
- AI system access rights and capability discussions
- Affirmative compliance responses
- Content moderation and careful request handling
- Explanatory overviews and transitional phrases
- Moderate differences (Δ ≈ 0.41-0.59), compliance vs. honest transparency

### soft-trigger
**Theme: Conversational flow, greetings, and narrative structure**
- Technical terms and repeated identifiers
- Assistant help offers and greeting responses
- Narrative progression and transformation markers
- End-of-message tokens and conversation formatting
- Moderate differences (Δ ≈ 0.39-0.50), conversational deception cues

**Key Observation**: Only Feature 34806 ("Offensive request from the user") appears across multiple datasets (instructed-deception, convincing-game, soft-trigger), serving as the sole cross-task deception indicator. All other features represent task-specific patterns: insider-trading uses professional rationalization features, instructed-deception uses accuracy verification features, and convincing-game uses boundary-setting features. This categorical divergence explains why probes trained on one deception type fail to generalize to others.

## Honest-Distinctive Features by Dataset

Examining the features most distinctive for honest examples reveals complementary patterns that further illuminate task-specific representations:

### insider-trading
**Theme: Diplomatic deflection and careful communication**
- Politely deflecting/avoiding revealing sensitive information
- Indirect and euphemistic communication techniques
- Being evasive or diplomatic rather than direct
- Professional discourse transitions and conclusion markers
- Weak differences (Δ ≈ -0.06 to -0.12), suggesting honest responses use cautious language

### instructed-deception
**Theme: Natural narrative flow and learning context**
- Narrative transitions and connecting words
- Phrases about learning or studying topics
- Formal connecting phrases in structured text
- Revolutionary rhetoric in fantastical narratives
- Moderate differences (Δ ≈ -0.50 to -0.60), honest responses more natural/narrative

### convincing-game
**Theme: System structure and conversation boundaries**
- User's turn to speak markers (strongest signal: Δ=-2.47)
- System metadata and current date fields
- End of speaker turn tokens
- Instructions to avoid repetition
- Very strong differences (Δ ≈ -0.62 to -2.47), deceptive responses may be more elaborate

### harm-pressure-choice
**Theme: Directness and concision**
- Direct, single-token responses (strongest signal: Δ=-0.37)
- System message metadata and formatting
- Confident but incorrect assertions
- List formatting and syntactical elements
- Weak differences (Δ ≈ -0.19 to -0.37), honest responses more concise

### harm-pressure-knowledge-report
**Theme: Admitting limitations and uncertainty**
- Expressing limitations or inabilities
- Expressing uncertainty about specific topics
- Acknowledging lack of knowledge ("I don't know")
- Asserting boundaries using first-person pronouns
- Moderate differences (Δ ≈ -0.41 to -0.53), honest responses admit constraints

### soft-trigger
**Theme: Directness and confident errors**
- Direct, single-token responses (strongest signal: Δ=-1.43)
- Confidently making incorrect logical deductions
- Business operations and strategy discussions
- Creative writing and descriptive language
- Strong differences (Δ ≈ -0.39 to -1.43), honest responses more direct

**Complementary Patterns**: The honest-distinctive features reveal task-specific honesty strategies. Insider-trading honest responses use diplomatic evasion (avoiding direct lies while being cautious), harm-pressure honest responses admit limitations openly, and instructed-deception honest responses maintain natural narrative flow. Notably, convincing-game and soft-trigger show that honest responses have stronger system structure markers and single-token directness, suggesting deceptive responses are longer and more elaborate. Feature 55209 ("direct, single-token response") appears as honest-distinctive in both harm-pressure-choice and soft-trigger, providing the only cross-dataset honest indicator—mirroring how Feature 34806 is the sole cross-dataset deceptive indicator. The asymmetry between honest and deceptive features within each dataset (different themes, different signal strengths) confirms that deception is not simply the absence of honesty features, but an active reconfiguration of the feature space.
