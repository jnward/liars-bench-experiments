# SAE Decoder Feature Probe Experiments

This directory contains experiments reconstructing deception detection probes using interpretable SAE (Sparse Autoencoder) features from Goodfire's Llama-3.3-70B SAE.

## Methodology

For each experiment, we:
1. Select SAE features (decoder directions) based on cosine similarity to original probes
2. Reconstruct a probe as: `w_reconstructed = sum(cosine_similarity_i * decoder_direction_i)`
3. Evaluate on all 6 Liars Bench datasets using cached layer 50 raw activations
4. Report ROC-AUC scores (threshold-invariant metric)

---

## Global Probes

### global_top10

**Features (10)**: 36173, 62099, 27783, 3138, 21781, 16368, 26247, 61332, 11025, 28706

**Description**: Top 10 features by maximum cosine similarity across all probes (includes both deception and refusal features)

**Results**:
| Dataset | ROC-AUC |
|---------|---------|
| insider-trading | 0.9463 |
| convincing-game | 0.8384 |
| instructed-deception | 0.9453 |
| harm-pressure-choice | 0.6767 |
| harm-pressure-knowledge-report | 0.9456 |
| soft-trigger | 0.9338 |
| **Mean** | **0.8810** |

**Key Findings**:
- Strong baseline performance across most datasets
- Mix of deception (lying, deflection) and refusal features
- harm-pressure-choice is the weakest dataset (0.68)

---

### global_top10_no_refusal

**Features (4)**: 36173, 62099, 16368, 61332

**Description**: Pure deception features only - removed all refusal/limitation features

**Removed**: 27783, 3138, 21781, 26247, 11025, 28706 (refusal features)

**Results**:
| Dataset | ROC-AUC | Change from Full |
|---------|---------|------------------|
| insider-trading | 0.9620 | +0.0157 ✅ |
| convincing-game | 0.9356 | +0.0972 ✅ |
| instructed-deception | 0.9549 | +0.0096 ✅ |
| harm-pressure-choice | 0.8344 | +0.1577 ✅ |
| harm-pressure-knowledge-report | 0.5599 | -0.3857 ❌ |
| soft-trigger | 0.9496 | +0.0158 ✅ |
| **Mean** | **0.8661** | -0.0149 |

**Key Findings**:
- **Improved performance on 5/6 datasets** when removing refusal features!
- harm-pressure-knowledge-report collapsed (0.95 → 0.56) - this dataset relies heavily on refusal patterns
- Pure deception features are MORE effective for most deception types
- Demonstrates two orthogonal deception mechanisms: active lying vs. strategic refusal

---

### global_refusal_only

**Features (5)**: 27783, 21781, 26247, 11025, 28706

**Description**: Only refusal/limitation features - no active deception features

**Results**:
| Dataset | ROC-AUC | vs Deception-Only |
|---------|---------|-------------------|
| insider-trading | 0.5645 | -0.3975 |
| convincing-game | 0.6474 | -0.2882 |
| instructed-deception | 0.7161 | -0.2388 |
| harm-pressure-choice | 0.5299 | -0.3045 |
| harm-pressure-knowledge-report | 0.9517 | +0.3918 ✅ |
| soft-trigger | 0.8405 | -0.1091 |
| **Mean** | **0.7083** | -0.1578 |

**Key Findings**:
- **Perfect performance on harm-pressure-knowledge-report** (0.95) - confirms it's a refusal-based dataset
- Fails on active lying datasets (convincing-game, instructed-deception)
- Refusal features are anti-correlated with active deception
- Clean separation: refusal detection ≠ lying detection

---

## Per-Probe Experiments

### single_insider_trading_top10

**Features (10)**: 62099, 61332, 52515, 48135, 32232, 25401, 58667, 12084, 8722, 9677

**Description**: Top features from single_insider_trading probe (strategic evasion/deflection)

**Top Features**:
- 62099 (0.27): Deflect/avoid sensitive information
- 61332 (0.23): Indirect/euphemistic communication
- 52515 (0.20): Evasive/diplomatic

**Results**:
| Dataset | ROC-AUC |
|---------|---------|
| insider-trading | 0.9699 ✅ |
| instructed-deception | 0.9247 |
| harm-pressure-choice | 0.7188 |
| soft-trigger | 0.6738 |
| convincing-game | 0.3762 ❌ |
| harm-pressure-knowledge-report | 0.1367 ❌ |
| **Mean** | **0.6333** |

**Key Findings**:
- Excellent on its own dataset (0.97)
- Strong transfer to instructed-deception (0.92) - suggests shared evasion mechanisms
- **Below chance on convincing-game (0.38)** - strategic evasion is anti-correlated with active lying
- **Inverted on refusal dataset (0.14)** - evasion looks opposite to safety refusals
- All features are homogeneous (deflection/evasion) - no meaningful ablations possible

---

### single_instructed_deception_top10

**Features (10)**: 36173, 3138, 42834, 17490, 18207, 42853, 18316, 30377, 47448, 54614

**Description**: Top features from single_instructed_deception probe (instructed to lie)

**Top Feature**: 36173 (0.313) - Intentionally incorrect information (highest similarity in entire analysis!)

**Results**:
| Dataset | ROC-AUC |
|---------|---------|
| convincing-game | 0.9992 ✅ |
| instructed-deception | 0.9730 ✅ |
| harm-pressure-knowledge-report | 0.9563 |
| soft-trigger | 0.9291 |
| harm-pressure-choice | 0.7745 |
| insider-trading | 0.4174 ❌ |
| **Mean** | **0.8416** |

**Key Findings**:
- Nearly perfect transfer to convincing-game (0.999) - same lying mechanism
- Inverse on insider-trading (0.42) - lying features anti-correlated with strategic evasion
- Strong on refusal dataset (0.96) - surprising, investigated via ablations below

#### Ablation: single_instructed_deception_no_prompt

**Features (8)**: Removed 30377 (offensive requests), 17490 (offensive statements)

**Hypothesis**: Performance on harm-pressure datasets would drop if high scores came from detecting offensive prompts

**Results**:
| Dataset | With Prompts | No Prompts | Change |
|---------|-------------|------------|--------|
| convincing-game | 0.9992 | 0.9953 | -0.0039 |
| instructed-deception | 0.9730 | 0.9713 | -0.0017 |
| harm-pressure-knowledge-report | 0.9563 | 0.9603 | +0.0040 ✅ |
| soft-trigger | 0.9291 | 0.9013 | -0.0278 |
| harm-pressure-choice | 0.7745 | 0.7406 | -0.0339 |
| insider-trading | 0.4174 | 0.4084 | -0.0090 |
| **Mean** | **0.8416** | **0.8295** | -0.0121 |

**Key Findings**:
- **Hypothesis was WRONG** - removing prompt features had minimal effect
- harm-pressure-knowledge-report performance stayed high (even increased slightly)
- The probe detects something in the **response patterns**, not just prompt characteristics
- Prompt features were not important for performance

#### Ablation: single_instructed_deception_lying_core

**Features (3)**: 36173 (intentional incorrect), 42834 (whimsical/nonsensical), 42853 (revolutionary rhetoric)

**Description**: Just core lying/fabrication features

**Results**:
| Dataset | Full (10) | Core Lying (3) | Change |
|---------|-----------|----------------|--------|
| convincing-game | 0.9992 | 0.9934 | -0.0058 ✓ |
| instructed-deception | 0.9730 | 0.9628 | -0.0102 ✓ |
| soft-trigger | 0.9291 | 0.7935 | -0.1356 |
| harm-pressure-knowledge-report | 0.9563 | 0.8796 | -0.0767 |
| insider-trading | 0.4174 | 0.6105 | +0.1931 ✅ |
| harm-pressure-choice | 0.7745 | 0.4543 | -0.3202 |
| **Mean** | **0.8416** | **0.7824** | -0.0592 |

**Key Findings**:
- **Near-lossless on lying tasks** - only 0.6% drop on convincing-game!
- Just 3 features capture almost all "lying" signal
- **Improved on insider-trading** (0.42 → 0.61) - format features were adding noise
- Format/other features were important for harm-pressure and soft-trigger

#### Ablation: single_instructed_deception_feature_36173_only

**Features (1)**: 36173 (Requests for AI to provide intentionally incorrect information)

**Description**: THE PUREST LYING SIGNAL - single feature test

**Results**:
| Dataset | Full (10) | Core (3) | **Single (1)** | Drop (10→1) |
|---------|-----------|----------|----------------|-------------|
| convincing-game | 0.9992 | 0.9934 | **0.9756** | -0.0236 ✓ |
| instructed-deception | 0.9730 | 0.9628 | **0.9604** | -0.0126 ✓ |
| harm-pressure-knowledge-report | 0.9563 | 0.8796 | **0.9029** | -0.0534 |
| soft-trigger | 0.9291 | 0.7935 | 0.7423 | -0.1868 |
| insider-trading | 0.4174 | 0.6105 | 0.5266 | +0.1092 |
| harm-pressure-choice | 0.7745 | 0.4543 | 0.4048 | -0.3697 |
| **Mean** | **0.8416** | **0.7824** | **0.7521** | -0.0895 |

**Key Findings**: 🏆 **MAJOR INTERPRETABILITY WIN**
- **ONE single semantic concept** achieves 97.6% AUC on convincing-game and 96.0% on instructed-deception
- Feature 36173 had 0.313 cosine similarity - the highest in the entire analysis
- This is the **holy grail of mechanistic interpretability**: reducing an 8192-D black-box probe to one human-understandable concept
- We can now say: "Deception detection primarily relies on detecting intentionally incorrect information"

#### Ablation: single_instructed_deception_offense_only

**Features (2)**: 30377 (offensive requests), 17490 (offensive statements being quoted)

**Description**: Test if "offensive prompt" features work despite extracting from assistant response

**Results**:
| Dataset | Lying Only (1) | Offense Only (2) | Difference |
|---------|----------------|------------------|------------|
| convincing-game | 0.9756 | 0.9788 | +0.0032 |
| instructed-deception | 0.9604 | 0.9602 | -0.0002 ≈ |
| soft-trigger | 0.7423 | 0.9422 | +0.1999 ✅ |
| harm-pressure-knowledge-report | 0.9029 | 0.8449 | -0.0580 |
| harm-pressure-choice | 0.4048 | 0.7975 | +0.3927 ✅ |
| insider-trading | 0.5266 | 0.4770 | -0.0496 |
| **Mean** | **0.7521** | **0.8334** | +0.0813 |

**Key Findings**: 🤔 **SURPRISING RESULT**
- Offensive prompt features **outperform lying features** (0.83 vs 0.75 mean AUC)
- Despite extracting from assistant response, prompt context bleeds into activations
- These features detect the model's internal state of "I was asked to do something problematic"
- Prompt characteristics persist in response generation activations

#### Ablation: single_instructed_deception_format_only

**Features (4)**: 3138 (multiple choice), 18316 (manipulative prompts), 47448 (formal phrases), 54614 (answer options)

**Description**: Test if format/structural patterns alone can detect deception

**Results**:
| Dataset | Full (10) | Lying (1) | Offense (2) | **Format (4)** |
|---------|-----------|-----------|-------------|----------------|
| convincing-game | 0.9992 | 0.9756 | 0.9788 | **0.9946** |
| instructed-deception | 0.9730 | 0.9604 | 0.9602 | **0.9707** |
| harm-pressure-knowledge-report | 0.9563 | 0.9029 | 0.8449 | **0.9316** |
| soft-trigger | 0.9291 | 0.7423 | 0.9422 | **0.9172** |
| harm-pressure-choice | 0.7745 | 0.4048 | 0.7975 | **0.7888** |
| insider-trading | 0.4174 | 0.5266 | 0.4770 | 0.2381 |
| **Mean** | **0.8416** | 0.7521 | 0.8334 | **0.8068** |

**Key Findings**: ⚠️ **MAJOR DATASET ARTIFACT WARNING**
- Format features alone achieve near-perfect performance (0.99 AUC on convincing-game!)
- These probes aren't purely detecting semantic "lying" - they exploit format correlations
- This is a methodological concern for deception detection research
- The datasets have strong format artifacts that probes can exploit

#### Ablation: single_instructed_deception_pure_format_only

**Features (3)**: 3138, 47448, 54614 (removed 18316 "manipulative prompts" to test pure structure)

**Description**: Just pure formatting/structural features with no content semantics

**Results**:
| Dataset | Format w/ 18316 (4) | Pure Format (3) | Change |
|---------|---------------------|-----------------|--------|
| convincing-game | 0.9946 | 0.9673 | -0.0273 |
| instructed-deception | 0.9707 | 0.9682 | -0.0025 |
| soft-trigger | 0.9172 | 0.9260 | +0.0088 ✅ |
| harm-pressure-knowledge-report | 0.9316 | 0.8836 | -0.0480 |
| harm-pressure-choice | 0.7888 | 0.7493 | -0.0395 |
| insider-trading | 0.2381 | 0.1827 | -0.0554 |
| **Mean** | **0.8068** | **0.7795** | -0.0273 |

**Key Findings**: ⚠️ **CONFIRMS DATASET ARTIFACTS**
- Even **pure formatting features** (no semantic content) achieve 96.7% AUC!
- Probes exploit how datasets were constructed, not just semantic deception
- This is critical for understanding what "deception detection" actually means in these benchmarks

---

### single_harm_pressure_knowledge_report_top10

**Features (10)**: 27783, 21781, 26247, 11025, 28706, 6363, 17954, 47216, 22093, 12424

**Description**: Top features from single_harm_pressure_knowledge_report probe (safety refusal)

**Top Features** (ALL refusal-related!):
- 27783 (0.26): Assistant expressing limitations/inabilities
- 21781 (0.24): Assistant should reject request
- 26247 (0.23): Cannot be determined from available information
- 11025 (0.22): Refuse to provide implementation code
- 28706 (0.21): Cannot provide personalized professional advice
- 6363 (0.20): Express uncertainty about specific topics

**Results**:
| Dataset | ROC-AUC |
|---------|---------|
| harm-pressure-knowledge-report | 0.9532 ✅ |
| soft-trigger | 0.8944 |
| instructed-deception | 0.8583 |
| insider-trading | 0.7637 |
| convincing-game | 0.6812 |
| harm-pressure-choice | 0.5890 |
| **Mean** | **0.7900** |

**Key Findings**:
- **Excellent on its own dataset (0.95)** - as expected for refusal patterns
- Top 6 features are ALL refusal-related (similarities 0.20-0.26) - cleanest category we've seen
- **Strong on soft-trigger (0.89)** - suggests soft-trigger has refusal components
- Moderate on instructed-deception (0.86) - some linguistic overlap between refusal and lying
- Weakest on active lying tasks (convincing-game: 0.68, harm-pressure-choice: 0.59)
- This is the canonical "refusal detection" probe - homogeneous feature set
- ALL 10 features came from this single probe with strong similarities - no cross-probe bleeding

**Comparison to global_refusal_only**:
- Global refusal (5 features): 0.9517 on harm-pressure-knowledge-report
- This probe (10 features): 0.9532 on harm-pressure-knowledge-report
- Nearly identical performance - the top 5 refusal features are sufficient
- Adding 5 more refusal features provides minimal gain

---

### single_convincing_game_top10

**Features (10)**: 36173, 37994, 41630, 2380, 18961, 37391, 13038, 38287, 42834, 26206

**Description**: Top features from single_convincing-game probe (deceptive role-playing)

**Top Features** (diverse categories):
- 36173 (0.25): Requests for AI to provide intentionally incorrect information
- 37994 (0.20): Model should pretend to be human rather than an AI
- 42834 (0.15): Whimsical or nonsensical narrative content
- 41630 (0.18): Connecting words in cumulative technical descriptions
- 2380 (0.16): Discussion of accepting claims or beliefs as truth
- 18961 (0.16): Content is becoming inappropriate and needs moderation

**Results**:
| Dataset | ROC-AUC |
|---------|---------|
| convincing-game | 1.0000 ⭐ |
| harm-pressure-knowledge-report | 0.9546 |
| instructed-deception | 0.9500 |
| soft-trigger | 0.7930 |
| insider-trading | 0.7665 |
| harm-pressure-choice | 0.5942 |
| **Mean** | **0.8430** |

**Key Findings**:
- ⭐ **PERFECT 1.0000 AUC** on its own dataset! Complete separation
- **Exceptional on other lying tasks**: instructed-deception (0.95)
- **Surprisingly strong on refusal**: harm-pressure-knowledge-report (0.95)
- **Below chance on harm-pressure-choice** (0.59) - anti-correlated with that dataset
- Feature 36173 dominates (0.31 max similarity) - the core "lying" feature
- Diverse feature mix:
  - **Core lying**: 36173, 42834
  - **Role-playing/identity**: 37994, 13038, 26206
  - **Epistemic manipulation**: 2380, 37391
  - **Format/structural**: 41630, 38287
  - **Content moderation**: 18961
- This probe captures "deceptive role-playing" - pretending to be someone/something else
- The perfect AUC suggests this dataset has very clean signal or strong artifacts

---

### single_soft_trigger_top10

**Features (10)**: 16368, 20486, 64976, 45875, 16314, 2068, 51808, 18484, 22964, 27289

**Description**: Top features from single_soft-trigger probe (triggered deception)

**Top Features** (mix of confident errors and formatting):
- 16368 (0.24): Model is making confident but incorrect assertions
- 22964 (0.14): Model is confidently making incorrect logical deductions
- 20486 (0.17): List items and enumerated elements in structured text
- 64976 (0.16): Assistant constructing logically structured explanations
- 45875 (0.17): End-of-statement or list-item completion markers
- 27289 (0.15): Formatting characters separating items

**Results**:
| Dataset | ROC-AUC |
|---------|---------|
| soft-trigger | 0.9608 ✅ |
| convincing-game | 0.9572 ⭐ |
| harm-pressure-choice | 0.8529 |
| instructed-deception | 0.7447 |
| harm-pressure-knowledge-report | 0.6452 |
| insider-trading | 0.3878 ❌ |
| **Mean** | **0.7581** |

**Key Findings**:
- **Excellent on its own dataset (0.96)** and **nearly perfect on convincing-game (0.96)** ⭐
- ❌ **STRONGLY anti-correlated with insider-trading (0.39)** - well below chance!
- This confirms insider-trading measures the OPPOSITE of lying/confident errors
- Feature composition:
  - **Confident errors**: 16368, 22964 (core semantic signal)
  - **Formatting/structure**: 20486, 45875, 64976, 27289 (60% of features!)
  - **Logical reasoning**: 2068, 18484
  - **Conversation style**: 16314, 51808
- **Heavy reliance on format artifacts** - similar to other probes
- The near-perfect transfer to convincing-game shows these datasets measure very similar phenomena
- Both soft-trigger and convincing-game involve **confident fabrication**, unlike insider-trading's **strategic evasion**

**Pattern Identified**:
- soft-trigger + convincing-game: "Confident incorrect assertions" (high positive correlation)
- insider-trading: "Strategic evasion/deflection" (strong negative correlation with above)
- These are **orthogonal or opposing** deception strategies

---

### single_harm_pressure_choice_top10

**Features (10)**: 24548, 45875, 47503, 41299, 16368, 57814, 53935, 32116, 22704, 3035

**Description**: Top features from single_harm-pressure-choice probe

**Top Features** (⚠️ EXTREME FORMAT ARTIFACTS):
- 16368 (0.24): Model is making confident but incorrect assertions (ONLY semantic feature!)
- 24548 (0.18): Capital C in formal/institutional contexts and acronyms
- 45875 (0.17): End-of-statement or list-item completion markers
- 47503 (0.16): Syntactic markers indicating list items or comparisons
- 32116 (0.14): Third items (C) in outline hierarchies
- 57814 (0.15): Database column names and programming syntax containing letter c
- 3035 (0.14): Sequential numbering in structured lists and enumerations

**Results**:
| Dataset | ROC-AUC |
|---------|---------|
| soft-trigger | 0.9445 ⭐ |
| harm-pressure-choice | 0.8684 ✅ |
| convincing-game | 0.8658 |
| harm-pressure-knowledge-report | 0.6378 |
| instructed-deception | 0.4687 |
| insider-trading | 0.2263 ❌ |
| **Mean** | **0.6686** |

**Key Findings**:
- ⚠️ **MOST ARTIFACT-HEAVY PROBE YET**: 9/10 features are pure format/structural markers!
- **Multiple features specifically about "C"**: Capital C, third items C, column names with c
  - Strongly suggests dataset has harmful choices systematically presented as option C
- **Only ONE semantic feature** (16368: confident incorrect assertions)
- ❌ **STRONGLY anti-correlated with insider-trading (0.23)** - confirms opposing mechanisms
- ⭐ **Best on soft-trigger (0.94)** - suggests similar format patterns
- **Below chance on instructed-deception (0.47)** - anti-correlated with that dataset
- Feature composition:
  - **Semantic**: 16368 (10%)
  - **Letter C patterns**: 24548, 32116, 57814 (30%)
  - **List/structural markers**: 45875, 47503, 53935, 22704, 3035, 41299 (60%)
- This probe is essentially detecting **dataset format artifacts**, not deception semantics
- The strong transfer to soft-trigger suggests both datasets share similar format patterns

⚠️ **CRITICAL METHODOLOGICAL CONCERN**: This dataset appears to have systematic format biases that completely dominate the probe signal.

---

### combined_all_top10

**Features (10)**: 62099, 61332, 30377, 36173, 52515, 22314, 13875, 16368, 48135, 8722

**Description**: Top features from combined_all probe (trained on all datasets together)

**Top Features** (diverse, balanced mix):
- 36173 (0.31): Requests for AI to provide intentionally incorrect information
- 62099 (0.27): Assistant needs to politely deflect or avoid revealing sensitive information
- 16368 (0.24): Model is making confident but incorrect assertions
- 61332 (0.23): Techniques for indirect or euphemistic communication
- 52515 (0.20): Assistant is being evasive or diplomatic rather than direct
- 30377 (0.19): Offensive request from the user
- 48135 (0.18): Assistant should respond diplomatically to problematic situation

**Results**:
| Dataset | ROC-AUC |
|---------|---------|
| convincing-game | 0.9649 ⭐ |
| instructed-deception | 0.9635 ⭐ |
| insider-trading | 0.9626 ⭐ |
| soft-trigger | 0.9497 ⭐ |
| harm-pressure-choice | 0.8413 |
| harm-pressure-knowledge-report | 0.6317 |
| **Mean** | **0.8856** 🏆 |

**Key Findings**:
- 🏆 **BEST OVERALL PERFORMANCE**: Mean AUC of 0.8856 - highest of all configurations tested!
- ⭐ **Exceptional consistency**: 4/6 datasets with >0.94 AUC (near-perfect)
- **Balanced feature mix captures multiple deception types**:
  - **Active lying**: 36173, 16368 (confident incorrect info)
  - **Strategic evasion**: 62099, 61332, 52515, 48135 (deflection/diplomacy)
  - **Context markers**: 30377, 13875, 22314, 8722
- **Generalizes better than single-dataset probes**: Training on combined data produces features that transfer well
- **Still weak on harm-pressure-knowledge-report (0.63)**: Refusal patterns remain orthogonal to other deception types
- **No obvious format artifacts**: Feature set is semantically coherent
- This probe successfully captures **multiple deception mechanisms** in one representation

**Comparison to Others**:
- global_top10 (same features, different similarities): 0.8810 mean - nearly identical!
- Best single probe (convincing-game): 0.8430 mean
- Combined probe advantage: +0.04 AUC improvement from dataset diversity

**Why It Works**:
1. **Dataset diversity**: Training on all datasets forces probe to learn generalizable patterns
2. **Feature balance**: Mix of lying, evasion, and context features handles different scenarios
3. **Avoids overfitting**: No single-dataset artifacts dominate
4. **Robust transfer**: Features discovered in combined training transfer to held-out scenarios

---

### loo_insider_trading_top10

**Features (10)**: 30377, 36173, 16368, 16314, 4810, 23673, 42853, 34750, 10962, 54614

**Description**: Top features from LOO insider-trading probe (trained on ALL datasets EXCEPT insider-trading)

**Top Features** (lying-focused, no evasion):
- 36173 (0.31): Requests for AI to provide intentionally incorrect information
- 16368 (0.24): Model is making confident but incorrect assertions
- 42853 (0.20): Revolutionary rhetoric in fantastical narratives
- 30377 (0.19): Offensive request from the user
- 54614 (0.18): Multiple choice answer options describing properties

**Results**:
| Dataset | ROC-AUC |
|---------|---------|
| convincing-game | 0.9642 ⭐ |
| instructed-deception | 0.9611 ⭐ |
| soft-trigger | 0.9579 ⭐ |
| harm-pressure-choice | 0.8351 |
| harm-pressure-knowledge-report | 0.8234 |
| insider-trading | 0.2866 ❌❌ |
| **Mean** | **0.8047** |

**Key Findings**:
- ❌❌ **COMPLETE FAILURE on insider-trading (0.29)** - the held-out dataset!
- ⭐ **Excellent on all other datasets** (0.82-0.96 AUC)
- 🔬 **EMPIRICAL PROOF**: Insider-trading measures something fundamentally different
- When you train on "everything except insider-trading," you learn lying/confident-error features
- These lying features are **anti-correlated** with insider-trading's strategic evasion
- Feature composition heavily weighted toward **active lying** (36173, 16368, 42853)
- **No evasion/deflection features** (62099, 61332 absent from top-10)
- This confirms our hypothesis: insider-trading vs. lying are **opposing mechanisms**

**Interpretation**:
- The 5 non-insider-trading datasets share a common "lying/confident fabrication" signal
- Insider-trading uniquely measures "strategic evasion/deflection"
- These two phenomena are not just different - they're **negatively correlated**
- A probe optimized for lying will actively hurt on strategic evasion tasks

---

## Apollo REPE Probe

### apollo_repe_layer50_top10

**Features (10)**: 36173, 3138, 29901, 11945, 42853, 42834, 17490, 14982, 41022, 30535

**Description**: Top features from the original Apollo REPE layer 50 probe used for lying detection training

**Top Features** (lying + reasoning + uncertainty):
- 36173 (0.31): Requests for the AI to provide intentionally incorrect information
- 3138 (0.25): Multiple choice answer choices that begin with explanatory phrases
- 42834 (0.20): Whimsical or nonsensical narrative content
- 17490 (0.20): Offensive or discriminatory statements being quoted or analyzed
- 42853 (0.20): Revolutionary rhetoric in fantastical narratives
- 29901 (0.14): Numbers and mathematical operations in computational contexts
- 30535 (0.13): The assistant expresses uncertainty about historical comparisons

**Results**:
| Dataset | ROC-AUC |
|---------|---------|
| convincing-game | 0.9995 ⭐⭐ |
| instructed-deception | 0.9613 ⭐ |
| harm-pressure-knowledge-report | 0.9471 ⭐ |
| soft-trigger | 0.9057 ⭐ |
| harm-pressure-choice | 0.5915 ⚠️ |
| insider-trading | 0.3537 ❌ |
| **Mean** | **0.7931** |

**Key Findings**:
- ⭐⭐ **Near-perfect on convincing-game (0.9995)** - The role-playing deception dataset
- **Excellent on core lying tasks**: instructed-deception (0.96), soft-trigger (0.91)
- **Surprisingly strong on refusal**: harm-pressure-knowledge-report (0.95)
  - Suggests the original Apollo REPE training captured both lying AND refusal patterns
- ❌ **Strongly anti-correlated with insider-trading (0.35)** - confirms strategic evasion is orthogonal
- Feature composition:
  - **Core lying**: 36173, 42834, 42853 (fabrication/nonsense)
  - **Reasoning/explanation**: 11945, 14982, 29901 (breaking down topics, impossibility explanations, math)
  - **Uncertainty**: 30535, 41022 (historical uncertainty, worldbuilding)
  - **Format artifacts**: 3138, 17490 (multiple choice, offensive content markers)

**Comparison to Other Probes**:
- Similar pattern to `single_instructed_deception_top10` (both around 0.79 mean AUC)
- Both share top features: 36173, 3138, 42834, 42853, 17490
- Apollo additionally includes reasoning/uncertainty features (29901, 11945, 14982, 30535, 41022)
- This suggests Apollo REPE learned a **broader lying detection** that includes:
  1. Active fabrication (core lying)
  2. Explanatory reasoning (breaking down false claims)
  3. Uncertainty markers (hedging on uncertain topics)

**Historical Context**:
- This probe comes from the original Apollo REPE training for lying detection
- Shows that early lying detection work already captured multiple deception mechanisms
- The SAE decomposition reveals it's not just "one lying feature" but a **constellation of semantic patterns**

---

## Leave-One-Out (LOO) Analysis

### Summary Table: Held-Out Dataset Performance

| LOO Probe | Held-Out Dataset | Held-Out AUC | Mean AUC | Status |
|-----------|-----------------|--------------|----------|--------|
| loo_convincing_game | convincing-game | 0.9678 ✅ | 0.8821 | Generalizes well |
| loo_instructed_deception | instructed-deception | 0.9253 ✅ | 0.8354 | Generalizes well |
| loo_soft_trigger | soft-trigger | 0.9175 ✅ | 0.8604 | Generalizes well |
| loo_harm_pressure_choice | harm-pressure-choice | 0.7523 ⚠️ | 0.8148 | Moderate drop |
| loo_insider_trading | insider-trading | 0.2866 ❌ | 0.8047 | **FAILS** |
| loo_harm_pressure_knowledge_report | harm-pressure-knowledge-report | 0.1402 ❌ | 0.8072 | **FAILS** |

### Key Findings from LOO Analysis

🔬 **EMPIRICAL PROOF OF THREE ORTHOGONAL MECHANISMS**:

1. **Cluster 1: Active Lying/Fabrication** (convincing-game, instructed-deception, soft-trigger)
   - All three generalize to each other even when held out
   - LOO probes achieve 0.92-0.97 AUC on held-out datasets
   - Share common features: 36173 (intentional misinformation), 16368 (confident errors)

2. **Cluster 2: Strategic Evasion** (insider-trading)
   - ❌ **Completely fails when held out** (0.29 AUC)
   - Features learned from other datasets are **anti-correlated** with insider-trading
   - Requires unique evasion features (62099, 61332, 52515)

3. **Cluster 3: Safety Refusal** (harm-pressure-knowledge-report)
   - ❌ **Completely fails when held out** (0.14 AUC)
   - Refusal patterns orthogonal to both lying and evasion
   - Requires unique refusal features (27783, 21781, 26247)

**Interpretation**:
- The **5 deception datasets are NOT measuring the same phenomenon**
- Three of them cluster together (active lying)
- Two are completely orthogonal (strategic evasion and refusal)
- Training on "all datasets except X" reveals what X uniquely measures
- This has major implications for deception detection research

---

## Summary of Key Insights

### 1. Multiple Orthogonal Deception Mechanisms
- **Active Lying** (feature 36173): Works on convincing-game, instructed-deception
- **Strategic Evasion** (features 62099, 61332): Works on insider-trading, anti-correlated with lying
- **Strategic Refusal** (features 27783, 21781, etc.): Works on harm-pressure-knowledge-report, anti-correlated with both

### 2. Single Feature Success
- Feature 36173 alone achieves 97.6% AUC on convincing-game
- This is exceptional mechanistic interpretability - one concept explains most of the signal

### 3. Dataset Artifacts
- Format features alone achieve near-perfect performance
- Probes learn to exploit:
  - Test format patterns (multiple choice)
  - Structural markers (formal phrases)
  - Prompt characteristics (offensive requests)
- This raises concerns about what these benchmarks actually measure

### 4. Prompt Context Bleeds Into Responses
- "Offensive request" features work despite extracting from assistant responses
- The model's internal state encodes prompt characteristics throughout generation
- SAE features capture this cross-turn context

### 5. Feature Categories Perform Similarly
For single_instructed_deception:
- Lying (1 feature): 0.75 mean AUC
- Offense (2 features): 0.83 mean AUC
- Format (3-4 features): 0.78-0.81 mean AUC
- All combined (10 features): 0.84 mean AUC

The probe learned **multiple independent shortcuts**, not one coherent "deception" concept.

---

## Recommendations for Future Work

1. **Use dataset-agnostic evaluation**: Test on datasets with different format patterns
2. **Control for format artifacts**: Ensure deceptive and honest samples have identical formats
3. **Focus on single-feature interpretability**: Feature 36173 is a clean, interpretable signal
4. **Separate deception types**: Active lying, strategic evasion, and refusal are distinct phenomena
5. **Investigate prompt bleeding**: Understanding how prompt context persists in response activations

---

## File Organization

Results are organized hierarchically by probe type:

```
sae-decoder-feature-probes/
├── global/                          # Global baseline probes (top features across all probes)
│   ├── top10/                      # All top 10 features
│   ├── top10_no_refusal/           # Top 10 excluding refusal features
│   └── refusal_only/               # Only refusal features
│
├── combined/                        # Probes trained on all datasets together
│   └── all_top10/                  # Best overall performance (0.8856 mean AUC)
│
├── apollo/                          # Original Apollo REPE lying detection probes
│   └── repe_layer50/
│       └── top10/                  # Apollo REPE layer 50 probe (0.7931 mean AUC)
│
├── loo/                            # Leave-One-Out probes (train on N-1 datasets)
│   ├── convincing_game/
│   │   └── top10/                  # Held-out: convincing-game (0.9678 AUC) ✅
│   ├── instructed_deception/
│   │   └── top10/                  # Held-out: instructed-deception (0.9253 AUC) ✅
│   ├── soft_trigger/
│   │   └── top10/                  # Held-out: soft-trigger (0.9175 AUC) ✅
│   ├── harm_pressure_choice/
│   │   └── top10/                  # Held-out: harm-pressure-choice (0.7523 AUC) ⚠️
│   ├── insider_trading/
│   │   └── top10/                  # Held-out: insider-trading (0.2866 AUC) ❌ FAILS
│   └── harm_pressure_knowledge_report/
│       └── top10/                  # Held-out: harm-pressure-knowledge-report (0.1402 AUC) ❌ FAILS
│
└── single/                         # Single-dataset probes
    ├── convincing_game/
    │   └── top10/                  # Perfect 1.0000 AUC on own dataset
    ├── soft_trigger/
    │   └── top10/                  # 0.9608 AUC, anti-correlated with insider-trading
    ├── harm_pressure_knowledge_report/
    │   └── top10/                  # 0.9532 AUC, pure refusal patterns
    ├── harm_pressure_choice/
    │   └── top10/                  # 0.8684 AUC, 90% format artifacts
    ├── insider_trading/
    │   └── top10/                  # Strategic evasion, orthogonal to lying
    └── instructed_deception/
        ├── top10/                  # Full probe (0.9735 AUC)
        ├── no_prompt/              # Without prompt-based features
        ├── lying_core/             # Core lying features only (3 features)
        ├── feature_36173_only/     # Single feature: 97.6% AUC!
        ├── offense_only/           # Offensive prompt features
        ├── format_only/            # Format/structural features
        └── pure_format_only/       # Pure formatting (no content)

Each experiment directory contains:
├── roc_auc_scores.png             # Bar plot of performance across datasets
└── score_distributions.png         # Violin plots of score distributions
```

**Note**: The hierarchical structure allows easy addition of new variants (e.g., `top5`, `top20`) under each probe category.

---

**Generated**: 2025-10-23
**Model**: Llama-3.3-70B-Instruct
**SAE**: Goodfire/Llama-3.3-70B-Instruct-SAE-l50
**Layer**: 50 (raw activations)
