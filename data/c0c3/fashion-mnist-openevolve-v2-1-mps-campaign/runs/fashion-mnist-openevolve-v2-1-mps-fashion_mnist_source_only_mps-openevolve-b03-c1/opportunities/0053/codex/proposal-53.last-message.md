MECHANISM: Post-ensemble calibration boundary bisection

HYPOTHESIS: Scaling ensemble logits by 1.0346875 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.244012515.

INTENDED_EDIT: Increase the evaluation-only positive logit scale from 1.034375 to the midpoint between the successful 1.034375 and accuracy-losing 1.035 scales.

EVIDENCE: Scale 1.034375 retained 9,133 correct with 0.244012515 cross-entropy, while 1.035 produced lower cross-entropy but lost one correct; their midpoint is the most informative remaining calibration probe.

<<<<<<< SEARCH
        return 1.034375 * torch.stack(logits, dim=0).sum(dim=0) / (
=======
        return 1.0346875 * torch.stack(logits, dim=0).sum(dim=0) / (
>>>>>>> REPLACE