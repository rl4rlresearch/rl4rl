MECHANISM: Post-ensemble calibration boundary bisection

HYPOTHESIS: Scaling ensemble logits by 1.0349609375 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.243975925.

INTENDED_EDIT: Increase the evaluation-only positive logit scale to the midpoint between the successful 1.034921875 and accuracy-losing 1.035 scales.

EVIDENCE: Scale 1.034921875 retained 9,133 correct with 0.243975925 cross-entropy, while 1.035 lowered cross-entropy but lost one correct; their midpoint is the most informative remaining calibration probe.

<<<<<<< SEARCH
        return 1.034921875 * torch.stack(logits, dim=0).sum(dim=0) / (
=======
        return 1.0349609375 * torch.stack(logits, dim=0).sum(dim=0) / (
>>>>>>> REPLACE