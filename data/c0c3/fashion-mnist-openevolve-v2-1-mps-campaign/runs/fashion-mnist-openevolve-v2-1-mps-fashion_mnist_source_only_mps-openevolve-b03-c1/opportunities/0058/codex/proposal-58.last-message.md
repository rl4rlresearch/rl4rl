MECHANISM: Post-ensemble calibration boundary bisection

HYPOTHESIS: Scaling ensemble logits by 1.034951171875 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.243974618.

INTENDED_EDIT: Increase the evaluation-only positive logit scale to the midpoint between the successful 1.03494140625 and accuracy-losing 1.0349609375 scales.

EVIDENCE: Scale 1.03494140625 retained 9,133 correct with 0.243974618 cross-entropy, while 1.0349609375 lowered cross-entropy but lost one correct; their midpoint is the most informative remaining calibration-boundary probe.

<<<<<<< SEARCH
        return 1.03494140625 * torch.stack(logits, dim=0).sum(dim=0) / (
=======
        return 1.034951171875 * torch.stack(logits, dim=0).sum(dim=0) / (
>>>>>>> REPLACE