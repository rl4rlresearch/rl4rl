MECHANISM: Longer-horizon exponential moving average

HYPOTHESIS: Increasing EMA decay from 0.99 to 0.995 will exceed 9,290 correct predictions by averaging roughly twice as many late-training updates, reducing parameter and BatchNorm-state noise without changing exposure or runtime materially.

INTENDED_EDIT: Lengthen the effective EMA window from approximately 100 to 200 optimizer steps while preserving the verified model, training augmentation, schedule, and evaluation ensemble.

EVIDENCE: Test-time ensembling previously improved validation correct from 9,138 to 9,167, while recent architecture and regularization changes all regressed from 9,290; refining the existing ensemble’s temporal component is therefore an informative low-risk change.

<<<<<<< SEARCH
            decay = 0.99
=======
            decay = 0.995
>>>>>>> REPLACE