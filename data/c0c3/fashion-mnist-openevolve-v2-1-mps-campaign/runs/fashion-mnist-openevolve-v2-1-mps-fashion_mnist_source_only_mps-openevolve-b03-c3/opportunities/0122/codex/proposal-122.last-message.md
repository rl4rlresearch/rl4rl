MECHANISM: Probability-space translation marginalization

HYPOTHESIS: Restoring beta2=0.96 and averaging translation predictions in probability space will exceed 9,283 correct predictions by limiting domination from overconfident atypical shifts while retaining the verified TTA kernel.

INTENDED_EDIT: Restore the best verified AdamW beta2 and replace logit averaging across translations with weighted softmax-probability averaging; preserve logit averaging across horizontal flips.

EVIDENCE: Beta2=0.96 produced the best 9,283-correct result, while nearby optimizer and augmentation changes regressed; full translation-kernel alignment lowered cross-entropy, motivating refinement of how the established translation views are combined.

<<<<<<< SEARCH
        logits = 0.36 * self._flip_average(images)
=======
        probabilities = 0.36 * self._flip_average(images).softmax(dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                logits = logits + weight * self._flip_average(shifted)
        return 1.253 * logits
=======
                probabilities = probabilities + weight * self._flip_average(
                    shifted
                ).softmax(dim=-1)
        return 1.253 * probabilities.clamp_min(1.0e-8).log()
>>>>>>> REPLACE

<<<<<<< SEARCH
        betas=(0.9, 0.95),
=======
        betas=(0.9, 0.96),
>>>>>>> REPLACE