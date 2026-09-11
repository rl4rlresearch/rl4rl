MECHANISM: Float32-adjacent fusion refinement

HYPOTHESIS: Moving exactly one representable float32 step below the best verified fusion weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Use complementary evaluation-fusion coefficients 0.49149906635284424 and 0.5085009336471558, retaining calibration and training behavior.

EVIDENCE: The best weight, 0.5085009765625, is bracketed by worse same-accuracy results at 0.50850048828125 and 0.50850146484375. Their cross-entropies place the interpolated minimum slightly below the center, motivating the immediately lower representable float32 coefficient.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.4914921875 * logits + 0.5085078125 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.49149906635284424 * logits
                + 0.5085009336471558 * flipped_logits
            )
>>>>>>> REPLACE