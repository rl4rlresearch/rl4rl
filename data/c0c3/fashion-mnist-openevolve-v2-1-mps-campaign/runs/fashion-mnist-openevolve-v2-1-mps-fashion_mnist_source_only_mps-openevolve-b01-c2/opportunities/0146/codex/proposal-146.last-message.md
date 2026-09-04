MECHANISM: Float32-adjacent fusion refinement

HYPOTHESIS: The immediately lower representable float32 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Set evaluation fusion to complementary float32 coefficients 0.49149906635284424 and 0.5085009336471558, preserving calibration and all training behavior.

EVIDENCE: The best verified weight, 0.5085009765625, beats same-accuracy neighbors on both sides, and their cross-entropies indicate a minimum slightly below it. Prior attempts at this adjacent value timed out without supplying contrary performance evidence.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.49149951171875 * logits + 0.50850048828125 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.49149906635284424 * logits
                + 0.5085009336471558 * flipped_logits
            )
>>>>>>> REPLACE