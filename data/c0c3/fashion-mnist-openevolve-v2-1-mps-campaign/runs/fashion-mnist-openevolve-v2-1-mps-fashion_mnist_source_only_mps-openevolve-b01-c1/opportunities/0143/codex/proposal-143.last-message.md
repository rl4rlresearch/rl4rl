MECHANISM: Soft-probability and plurality-vote TTA hybrid

HYPOTHESIS: Blending 10% equal hard votes into the successful arithmetic probability ensemble will exceed 9,266 correct predictions by reducing residual sensitivity to differing confidence magnitudes across transformed views.

INTENDED_EDIT: Keep training unchanged and add a small one-hot vote component to each validation-view probability vector before the existing spatial aggregation.

EVIDENCE: Arithmetic probability pooling improved validation correct from 9,265 to 9,266, while aligning the training loss to probability pooling regressed to 9,229; this motivates refining only inference aggregation toward confidence-independent voting.

<<<<<<< SEARCH
                probabilities = F.softmax(logits, dim=-1)
                original_probs, flipped_probs = probabilities.chunk(2, dim=0)
=======
                probabilities = F.softmax(logits, dim=-1)
                hard_votes = F.one_hot(
                    logits.argmax(dim=-1), num_classes=10
                ).to(probabilities.dtype)
                probabilities = 0.9 * probabilities + 0.1 * hard_votes
                original_probs, flipped_probs = probabilities.chunk(2, dim=0)
>>>>>>> REPLACE