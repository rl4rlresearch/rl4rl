MECHANISM: Interpolated target smoothing

HYPOTHESIS: Label smoothing of 0.023 will exceed 9,279 correct predictions by probing near the empirical accuracy maximum implied by the verified 0.01, 0.02, and 0.03 results.

INTENDED_EDIT: Increase label smoothing from 0.02 to 0.023 and restore the best-verified evaluation scale of 1.17817.

EVIDENCE: Smoothing 0.01 produced 9,258 correct, 0.02 produced 9,279, and 0.03 produced 9,274; quadratic interpolation of those results places the estimated maximum near 0.023, while scale 1.17817 has the best verified baseline cross-entropy.

<<<<<<< SEARCH
        if not self.training:
            logits = 1.17816 * logits
=======
        if not self.training:
            logits = 1.17817 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.023)
>>>>>>> REPLACE