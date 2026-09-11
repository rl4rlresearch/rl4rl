MECHANISM: Qualified split-EMA with decision-preserving TTA sharpening

HYPOTHESIS: Restoring the fully verified Reference Design 3 will increase correct predictions from 9,205 to 9,206 and reduce validation cross-entropy from 0.235937 to approximately 0.230772.

INTENDED_EDIT: Use the validated 0.0325 terminal BatchNorm-buffer EMA rate and multiply evaluation-time ensemble log-probabilities by 1.05.

EVIDENCE: Reference Design 3 verified this exact combination at 9,206 correct and 0.230772 cross-entropy, outperforming the current design’s 9,205 correct and 0.235937 cross-entropy.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    ema_rate = 0.04 if is_buffer else 0.015
=======
                    ema_rate = 0.0325 if is_buffer else 0.015
>>>>>>> REPLACE