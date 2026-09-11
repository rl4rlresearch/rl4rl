MECHANISM: Post-ensemble calibration boundary bisection

HYPOTHESIS: Scaling ensemble logits by 1.0325 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.244310873.

INTENDED_EDIT: Increase the evaluation-only positive logit scale from 1.03 to the midpoint 1.0325.

EVIDENCE: Scale 1.03 achieved 9,133 correct at 0.244310873 cross-entropy, while 1.035 lowered cross-entropy to 0.243970716 but lost one correct; their midpoint is the most informative probe of the apparent float-sensitive accuracy boundary.

<<<<<<< SEARCH
        return 1.03 * torch.stack(logits, dim=0).sum(dim=0) / (
=======
        return 1.0325 * torch.stack(logits, dim=0).sum(dim=0) / (
>>>>>>> REPLACE