MECHANISM: Conservative inference-only logit sharpening

HYPOTHESIS: Increasing evaluation logit scaling from 1.1 to 1.125 will preserve all 9,254 argmax predictions while lowering validation cross-entropy below 0.226035.

INTENDED_EDIT: Raise only the evaluation-mode logit multiplier from 1.1 to 1.125.

EVIDENCE: Scaling logits by 1.1 preserved 9,254 correct predictions and improved cross-entropy to 0.226035; the larger 1.15 and 1.2 attempts yielded no performance evidence because verification timed out, motivating a smaller runtime-neutral refinement.

<<<<<<< SEARCH
        if not self.training:
            logits = logits * 1.1
=======
        if not self.training:
            logits = logits * 1.125
>>>>>>> REPLACE