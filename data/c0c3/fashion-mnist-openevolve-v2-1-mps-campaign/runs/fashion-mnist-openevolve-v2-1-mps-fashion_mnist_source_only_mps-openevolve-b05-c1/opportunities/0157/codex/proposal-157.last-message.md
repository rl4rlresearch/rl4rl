MECHANISM: Longer-horizon late-training parameter EMA

HYPOTHESIS: Increasing the EMA decay cap from 0.98 to 0.99 will exceed 9,360 correct predictions by averaging a broader portion of the low-learning-rate trajectory while preserving the verified BatchNorm statistics and architecture.

INTENDED_EDIT: Extend only the parameter-averaging horizon; leave training, BatchNorm buffers, augmentation, TTA, and calibration unchanged.

EVIDENCE: Architecture changes fell to 9,326–9,337 correct, focal reweighting fell to 9,331, and averaging BatchNorm state fell to 9,351, motivating a minimal ranking-focused refinement of the otherwise successful parameter EMA.

<<<<<<< SEARCH
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
=======
            ema_decay = min(0.99, (updates + 1.0) / (updates + 10.0))
>>>>>>> REPLACE