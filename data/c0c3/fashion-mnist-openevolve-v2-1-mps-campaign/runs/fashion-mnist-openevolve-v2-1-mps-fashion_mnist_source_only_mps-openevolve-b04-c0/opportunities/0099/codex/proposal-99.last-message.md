MECHANISM: Accuracy-constrained canonical-view weight refinement

HYPOTHESIS: A 53.75% original / 46.25% flipped weighting will preserve all 9,359 correct predictions while lowering validation cross-entropy below 0.184717472076416.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 52.5% canonical weighting toward the lower-cross-entropy 55% weighting, retaining the established calibration scale.

EVIDENCE: The 52.5% weighting preserved 9,359 correct with 0.184717472076416 cross-entropy, while 55% reduced cross-entropy to 0.18461807746887207 but lost one correct prediction; their midpoint efficiently probes the accuracy boundary.

<<<<<<< SEARCH
        return 1.226016 * (
            0.525 * logits + 0.475 * flipped_logits
        )
=======
        return 1.226016 * (
            0.5375 * logits + 0.4625 * flipped_logits
        )
>>>>>>> REPLACE