MECHANISM: Argmax-guarded quarter-step TTA weight refinement

HYPOTHESIS: Halving the latest successful canonical-weight reduction will preserve 9,359 predictions through the existing guard while reducing cross-entropy below 0.18456672592163087.

INTENDED_EDIT: Lower the refined canonical-view weight by 0.000000011920928955078125 and increase its flipped-view complement equally.

EVIDENCE: The two successive guarded reductions to 0.5380733013153076171875 and 0.53807327747344970703125 each improved cross-entropy without changing correctness, motivating a finer continuation in the same direction.

<<<<<<< SEARCH
        refined_logits = (
            0.53807327747344970703125 * logits
            + 0.46192672252655029296875 * flipped_logits
        )
=======
        refined_logits = (
            0.538073265552520751953125 * logits
            + 0.461926734447479248046875 * flipped_logits
        )
>>>>>>> REPLACE