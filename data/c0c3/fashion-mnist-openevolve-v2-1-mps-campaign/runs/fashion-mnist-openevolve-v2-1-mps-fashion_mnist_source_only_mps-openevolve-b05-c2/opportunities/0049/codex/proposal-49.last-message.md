MECHANISM: Order-locked midpoint temperature sharpening

HYPOTHESIS: Sharpening from 16% to 24% while anchoring class ordering to the proven 84%-denominator logits will retain 9,192 correct predictions and reduce cross-entropy below 0.22403448944091797.

INTENDED_EDIT: Center the current logits on their per-image maximum, then apply the 21/19 scale corresponding to a 76%-of-normalization denominator.

EVIDENCE: The 84% denominator achieved 9,192 correct at 0.22403448944091797 cross-entropy, while 68% lost one prediction; 76% is the informative midpoint, and max-centering before positive scaling preserves the 84% logits’ argmax.

<<<<<<< SEARCH
        return logit_sum / 13.2124453125
=======
        logits = logit_sum / 13.2124453125
        logits = logits - logits.amax(dim=1, keepdim=True)
        return logits * (21.0 / 19.0)
>>>>>>> REPLACE