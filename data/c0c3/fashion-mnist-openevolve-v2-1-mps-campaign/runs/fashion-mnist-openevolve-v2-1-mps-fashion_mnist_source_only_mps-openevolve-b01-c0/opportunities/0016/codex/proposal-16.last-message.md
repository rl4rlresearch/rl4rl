MECHANISM: Dense-head dropout ablation

HYPOTHESIS: Removing dense-head dropout will exceed 9,290 correct predictions by allowing the near-ceiling 58-unit bottleneck to use its full capacity during the limited two-pass training exposure.

INTENDED_EDIT: Replace the classifier’s 10% dropout with an identity operation while preserving its architecture, parameter count, augmentation, optimizer, schedule, and EMA ensemble.

EVIDENCE: Expanding the dense bottleneck from 48 to 58 units improved validation correct from 9,265 to 9,290, while convolutional reallocation, channel gating, and projected widening regressed; this indicates that preserving and more fully optimizing the successful dense representation is the most relevant remaining lever.

<<<<<<< SEARCH
            nn.Dropout(p=0.1),
=======
            nn.Identity(),
>>>>>>> REPLACE