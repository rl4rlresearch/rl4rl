MECHANISM: Increased target smoothing

HYPOTHESIS: Raising label smoothing from 0.02 to 0.03 will increase validation_correct above 9,279 by extending the accuracy-improving direction observed between 0.01 and 0.02.

INTENDED_EDIT: Increase training label smoothing to 0.03 and restore the best-verified evaluation scale of 1.17817.

EVIDENCE: Reducing smoothing to 0.01 lowered validation_correct from 9,279 to 9,258, making a symmetric higher-side probe the most informative next test; scale 1.17817 has the best verified baseline cross-entropy.

<<<<<<< SEARCH
        if not self.training:
            logits = 1.178165 * logits
=======
        if not self.training:
            logits = 1.17817 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
>>>>>>> REPLACE