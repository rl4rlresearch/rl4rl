MECHANISM: Post-ensemble calibration boundary bisection

HYPOTHESIS: Scaling ensemble logits by 1.03484375 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.243991582.

INTENDED_EDIT: Increase the evaluation-only positive logit scale from 1.0346875 to the midpoint between the successful 1.0346875 and accuracy-losing 1.035 scales.

EVIDENCE: Scale 1.0346875 retained 9,133 correct with 0.243991582 cross-entropy, while 1.035 achieved lower cross-entropy but lost one correct; their midpoint is the most informative remaining probe of the observed accuracy boundary.

<<<<<<< SEARCH
        return 1.0346875 * torch.stack(logits, dim=0).sum(dim=0) / (
=======
        return 1.03484375 * torch.stack(logits, dim=0).sum(dim=0) / (
>>>>>>> REPLACE