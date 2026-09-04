MECHANISM: Mild post-ensemble temperature calibration

HYPOTHESIS: Dividing evaluation logits by 1.03 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.

INTENDED_EDIT: Apply temperature 1.03 to the existing sequential 12-weight TTA ensemble without changing training or evaluation workload.

EVIDENCE: The current design has the best observed correct count; stronger tail averaging reduced cross-entropy but changed decision boundaries, while prior temperature attempts timed out and therefore provide no accuracy or calibration evidence.

<<<<<<< SEARCH
        return ensemble - math.log(12.0)
=======
        return (ensemble - math.log(12.0)) / 1.03
>>>>>>> REPLACE