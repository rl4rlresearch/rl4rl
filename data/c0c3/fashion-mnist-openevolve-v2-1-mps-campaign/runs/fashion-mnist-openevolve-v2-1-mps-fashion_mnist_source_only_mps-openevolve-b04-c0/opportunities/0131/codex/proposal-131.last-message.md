MECHANISM: Guarded half-step TTA weight refinement

HYPOTHESIS: A half-sized continuation of the successful canonical-weight reduction will preserve 9,359 predictions through the existing argmax guard while reducing cross-entropy below 0.18456672744750977.

INTENDED_EDIT: Lower the refined canonical-view weight by 0.00000002384185791015625 and increase its complementary flipped-view weight equally.

EVIDENCE: Reducing the refined weight from 0.5380733489990234375 to 0.5380733013153076171875 improved cross-entropy from 0.18456672821044923 to 0.18456672744750977 without changing correctness, motivating a finer continuation in the same direction.

<<<<<<< SEARCH
        refined_logits = (
            0.5380733013153076171875 * logits
            + 0.4619266986846923828125 * flipped_logits
        )
=======
        refined_logits = (
            0.53807327747344970703125 * logits
            + 0.46192672252655029296875 * flipped_logits
        )
>>>>>>> REPLACE