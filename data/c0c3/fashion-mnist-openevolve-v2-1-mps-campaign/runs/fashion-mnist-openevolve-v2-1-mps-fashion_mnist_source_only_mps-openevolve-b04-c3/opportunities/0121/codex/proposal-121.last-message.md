MECHANISM: Longer-horizon exponential moving average

HYPOTHESIS: Increasing EMA decay from 0.99 to 0.995 will reduce update noise over an effective horizon of roughly 200 steps and exceed 9,247 correct predictions, or tie while lowering validation cross-entropy.

INTENDED_EDIT: Change only the EMA decay; architecture, training budget, optimizer, augmentation, and inference calibration remain unchanged.

EVIDENCE: The 0.99 EMA improved validation_correct from 9,243 to 9,247 with no runtime penalty, while heavier augmentation and architectural experiments timed out; extending the successful averaging mechanism is the lowest-cost ranking-changing test.

<<<<<<< SEARCH
        ema_model.update_ema(decay=0.99)
=======
        ema_model.update_ema(decay=0.995)
>>>>>>> REPLACE