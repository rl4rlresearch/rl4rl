MECHANISM: Best-observed BatchNorm EMA with decision-preserving TTA sharpening

HYPOTHESIS: Combining the reported 0.03 BatchNorm-buffer EMA rate with 1.05× evaluation sharpening will retain 9,206 correct predictions while reducing cross-entropy below 0.230772.

INTENDED_EDIT: Change terminal BatchNorm-buffer EMA from 0.04 to 0.03 and sharpen the final ensemble log-probabilities by 1.05.

EVIDENCE: A 0.03 buffer rate reportedly achieved 9,206 correct with lower unsharpened cross-entropy than 0.0325, while 1.05× sharpening preserved 9,206 correct and reduced cross-entropy from 0.235889 to 0.230772.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    ema_rate = 0.04 if is_buffer else 0.015
=======
                    ema_rate = 0.03 if is_buffer else 0.015
>>>>>>> REPLACE