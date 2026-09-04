MECHANISM: Symmetric order-preserving temperature counterprobe

HYPOTHESIS: An effective denominator of 76.511644% will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.

INTENDED_EDIT: Restore the proven 84%-denominator ensemble logits, max-center them, and test a temperature equally far below 76.512% as the unsuccessful 76.512356% probe was above it.

EVIDENCE: The 76.512% denominator achieved the best observed cross-entropy, 0.22237433319091796, while 76.512356% was slightly worse; the symmetric lower-side probe is the most direct local refinement.

<<<<<<< SEARCH
        return logit_sum / 15.7291015625
=======
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits * (84.0 / 76.511644)
>>>>>>> REPLACE