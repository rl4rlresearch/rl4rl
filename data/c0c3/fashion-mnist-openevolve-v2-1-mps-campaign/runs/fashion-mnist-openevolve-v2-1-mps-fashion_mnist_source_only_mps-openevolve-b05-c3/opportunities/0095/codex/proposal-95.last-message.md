MECHANISM: Decision-preserving intermediate TTA temperature sharpening

HYPOTHESIS: Restoring the qualified 0.0325 BatchNorm-buffer EMA and scaling ensemble log-probabilities by 1.075 will retain 9,206 correct predictions while reducing validation cross-entropy below 0.230772.

INTENDED_EDIT: Use the strongest qualified split-EMA training configuration and increase evaluation-only sharpening from 1.05× to 1.075× without changing predicted classes.

EVIDENCE: The 1.05× design retained 9,206 correct and reduced cross-entropy from 0.235889 to 0.230772; prior 1.075× attempts only timed out and supplied no performance evidence against the calibration change.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.075)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    ema_rate = 0.04 if is_buffer else 0.015
=======
                    ema_rate = 0.0325 if is_buffer else 0.015
>>>>>>> REPLACE