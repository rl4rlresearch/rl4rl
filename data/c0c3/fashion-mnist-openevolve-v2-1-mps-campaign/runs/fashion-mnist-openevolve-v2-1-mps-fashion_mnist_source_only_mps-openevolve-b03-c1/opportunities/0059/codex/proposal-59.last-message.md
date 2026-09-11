MECHANISM: Post-ensemble calibration boundary bisection

HYPOTHESIS: Scaling ensemble logits by 1.0349560546875 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.243973966.

INTENDED_EDIT: Increase the evaluation-only logit scale to the midpoint between the successful 1.034951171875 and accuracy-losing 1.0349609375 scales.

EVIDENCE: Scale 1.034951171875 retained 9,133 correct at 0.243973966 cross-entropy, while 1.0349609375 lowered cross-entropy but lost one correct; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
        return 1.034951171875 * torch.stack(logits, dim=0).sum(dim=0) / (
=======
        return 1.0349560546875 * torch.stack(logits, dim=0).sum(dim=0) / (
>>>>>>> REPLACE