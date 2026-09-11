MECHANISM: Float32-adjacent fusion refinement

HYPOTHESIS: Moving one representable float32 step below the best verified flipped-view weight will retain 9,331 correct predictions and lower validation cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Replace the current evaluation-fusion coefficients with complementary weights 0.49149906635284424 and 0.5085009336471558 while preserving calibration and training behavior.

EVIDENCE: The best verified weight, 0.5085009765625, outperformed same-accuracy neighbors on both sides; interpolation favors a slightly lower weight, and the sole prior verification of this adjacent value timed out without contrary performance evidence.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.49146875 * logits + 0.50853125 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.49149906635284424 * logits
                + 0.5085009336471558 * flipped_logits
            )
>>>>>>> REPLACE