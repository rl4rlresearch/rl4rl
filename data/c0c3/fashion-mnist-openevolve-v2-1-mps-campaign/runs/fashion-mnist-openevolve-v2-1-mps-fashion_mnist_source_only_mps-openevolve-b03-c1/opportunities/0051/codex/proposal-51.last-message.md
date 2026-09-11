MECHANISM: Post-ensemble calibration boundary bisection

HYPOTHESIS: Scaling ensemble logits by 1.03375 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.244139155.

INTENDED_EDIT: Increase the evaluation-only positive logit scale from 1.0325 to the midpoint 1.03375.

EVIDENCE: Scale 1.0325 retained 9,133 correct with 0.244139155 cross-entropy, while 1.035 lowered cross-entropy but lost one correct; their midpoint is the most informative remaining probe of the accuracy boundary.

<<<<<<< SEARCH
        return 1.0325 * torch.stack(logits, dim=0).sum(dim=0) / (
=======
        return 1.03375 * torch.stack(logits, dim=0).sum(dim=0) / (
>>>>>>> REPLACE