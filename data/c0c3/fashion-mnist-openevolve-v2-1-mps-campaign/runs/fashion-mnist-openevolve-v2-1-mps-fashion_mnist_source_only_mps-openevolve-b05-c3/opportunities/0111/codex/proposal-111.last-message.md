MECHANISM: Qualified split-EMA with decision-preserving TTA sharpening

HYPOTHESIS: Restoring the fully verified Reference Design 3 will increase validation correctness from 9,205 to approximately 9,206 and reduce cross-entropy from 0.235937 to approximately 0.230772.

INTENDED_EDIT: Set the terminal BatchNorm-buffer EMA rate to 0.0325 and sharpen evaluation-time ensemble log-probabilities by 1.05×.

EVIDENCE: Reference Design 3 verified this exact combination at 9,206 correct and 0.230772 cross-entropy, the strongest completed validation_score; subsequent attempts produced no completed contrary evidence.

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