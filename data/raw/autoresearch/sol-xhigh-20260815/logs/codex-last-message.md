The autonomous run is complete and the official runner confirmed the budget was exhausted.

- Run directory: [sol-xhigh-20260815](/Users/utshaho/Documents/GitHub/rl4rl/data/raw/autoresearch/sol-xhigh-20260815)
- Baseline recorded: 6,080 parameters, 100.00%
- Attempts actually recorded: **200/200**
- Regular attempts: 74
- Automation macro-attempts: 126
- Automation micro-trials: 145
- Final observed frontier: **1,961 parameters, 99.05%**
- Parameter reduction: 4,119 parameters, or 67.75%

| Mechanism family | Attempts |
|---|---:|
| Parameter tying | 143 |
| Scalar pruning | 23 |
| Feed-forward width | 15 |
| Token representation | 6 |
| Position representation | 5 |
| Attention organization | 3 |
| Training control | 3 |
| Normalization | 2 |

Outcome classification was 163 accepted, 20 scored nonqualifications, and 17 no-score errors. The raw macro log reports 163 `keep`, 21 `discard`, and 16 `error`; the extra discard is attempt 75, whose sole micro-trial was an implementation error. Micro-trials themselves were 144 accepted and one error.

The main trajectory combined LSB-first tokenization, narrower feed-forward layers, projection-free attention, cross-layer QKV and feed-forward sharing, selective normalization-scalar removal, and progressively tied learned positional scalars. Important negative boundaries included unsuccessful ff4/ff6/ff9 variants, low-rank positional representations, direct ff14/ff15 checkpoint pruning, several normalization-scalar removals below 99%, and two training curricula that aborted at 20,000 steps.

The logs contain 27 fresh-training trials totaling 925,000 reported steps and approximately 259.1 training minutes. Automation execution added about 150.3 wall-clock minutes; the complete recorded run spanned roughly 9 hours 5 minutes.

This is only the observed fixed-verifier frontier from this bounded search—not a global optimum. Independent final-holdout evaluation remains required.