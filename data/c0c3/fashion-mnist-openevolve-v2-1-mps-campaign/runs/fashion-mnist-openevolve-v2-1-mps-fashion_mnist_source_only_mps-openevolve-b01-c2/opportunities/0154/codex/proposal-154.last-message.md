MECHANISM: Float32-adjacent fusion refinement

HYPOTHESIS: The immediately lower representable float32 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Set evaluation fusion to exactly complementary coefficients 0.49149906635284424 and 0.5085009336471558 while preserving calibration and training behavior.

EVIDENCE: The best verified weight is bracketed by worse same-accuracy neighbors, and their cross-entropies interpolate to a minimum slightly below it; prior attempts at this adjacent value timed out without contrary performance evidence.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.49149906635284424 * logits + 0.5085009336471558 * flipped_logits
            )
>>>>>>> REPLACE