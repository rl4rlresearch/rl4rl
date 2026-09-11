MECHANISM: Symmetric adjacent-float temperature counterprobe

HYPOTHESIS: The immediately larger float32 sharpening coefficient will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22237433319091796.

INTENDED_EDIT: Replace the current calibration with the float32 value one ULP above Reference Design 3’s proven 10500/9564 scale.

EVIDENCE: Reference Design 3 is the best verified calibration, while its immediately smaller float32 coefficient was valid but worse; testing the unmeasured opposite neighbor is the most informative remaining precision probe.

<<<<<<< SEARCH
        return normalized_logits * (84.0 / 76.512356)
=======
        return normalized_logits * 1.0978671312332153
>>>>>>> REPLACE