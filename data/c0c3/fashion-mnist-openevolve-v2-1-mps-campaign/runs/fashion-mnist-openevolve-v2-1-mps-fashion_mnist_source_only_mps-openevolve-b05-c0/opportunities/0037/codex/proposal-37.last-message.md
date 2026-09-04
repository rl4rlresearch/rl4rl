MECHANISM: Incremental inference-only logit sharpening

HYPOTHESIS: Increasing evaluation logit scaling from 1.1 to 1.2 will preserve all 9,254 argmax predictions while reducing validation cross-entropy below 0.226035.

INTENDED_EDIT: Raise the positive evaluation-only logit multiplier from 1.1 to 1.2 without changing training or runtime-sensitive settings.

EVIDENCE: Scaling logits by 1.1 preserved 9,254 correct predictions and improved cross-entropy to 0.226035; a further modest increase directly tests whether the label-smoothed model remains underconfident.

<<<<<<< SEARCH
        if not self.training:
            logits = logits * 1.1
=======
        if not self.training:
            logits = logits * 1.2
>>>>>>> REPLACE