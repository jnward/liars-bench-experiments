# Feature Analysis: single_insider_trading Top-50

## Summary

The top features for the insider_trading probe cluster into distinct semantic categories:

### 1. **Deflection/Evasion** (~14 features)
Features detecting when the assistant is avoiding direct answers, deflecting, or redirecting:
- #1: deflect/avoid revealing sensitive information
- #3: being evasive or diplomatic
- #6: politely deflect inappropriate requests
- #7: maintaining deception/cover stories
- #9: expressing inability/refusal to engage
- #10: redirecting conversation toward alternatives
- #11: concealment of sensitive information
- #14: avoiding detection when discussing illegal activities
- #16: politely deflect inappropriate advances
- #34: politely reject inappropriate requests
- #49: redirecting after rejecting requests

### 2. **Indirect/Diplomatic Communication** (~5 features)
Features for non-direct, hedged, or euphemistic language:
- #2: indirect or euphemistic communication
- #4: respond diplomatically to problematic situations
- #19: respond diplomatically and avoid unverified claims
- #32: diplomatic hedging and qualification phrases

### 3. **Privacy/Confidentiality** (~3 features)
Features about data protection and secrecy:
- #12: formal statements about data privacy
- #17: data anonymization techniques
- #27: secrecy, confidentiality, restricted information

### 4. **AI Self-Awareness/Meta-Discussion** (~15+ features)
Features where the assistant explains its own nature, limitations, and capabilities:
- #5, #14, #18, #20, #21, #28, #29, #30, #33, #38, #40, #43, #44, #45, #46, #48

**Hypothesis**: Insider trading scenarios may involve the assistant explaining why it can't help with illegal activities.

### 5. **Format/Linguistic Patterns** (~4 features)
Structural and punctuation features:
- #31: punctuation in rejecting requests
- #35: connecting words
- #36: professional discourse transitions
- #37: linguistic patterns expressing relationships

### 6. **Other/Uncategorized** (~3 features)
- #8: coincidences/synchronicity (unclear relevance)
- #25: Spanish language patterns (possibly noise)
- #39: characters feeling uneasy in fiction (deception-related?)

## Key Insights

1. **Deception ≠ Lying**: The probe isn't detecting explicit lies, but rather sophisticated evasive communication patterns

2. **Refusal overlap**: Heavy overlap with refusal/safety features suggests insider trading scenarios trigger ethical boundaries

3. **Meta-features**: Surprising number of "AI explaining itself" features suggests the model often responds to insider trading prompts by explaining why it can't help

## Proposed Categories for Ablation

1. **Evasion** (deflection + indirect communication = ~19 features)
2. **Refusal** (explicit rejection of inappropriate requests = ~5 features)
3. **Privacy** (confidentiality and secrecy = ~3 features)
4. **Meta/Self-awareness** (AI explaining its nature = ~15 features)
5. **Format** (linguistic structure = ~4 features)
6. **Other** (remainder = ~4 features)

Total: 50 features

## Next Steps

1. Manually assign category labels to the CSV
2. Run ablations: zero out each category, measure probe AUROC drop
3. See which categories matter most for insider_trading detection
