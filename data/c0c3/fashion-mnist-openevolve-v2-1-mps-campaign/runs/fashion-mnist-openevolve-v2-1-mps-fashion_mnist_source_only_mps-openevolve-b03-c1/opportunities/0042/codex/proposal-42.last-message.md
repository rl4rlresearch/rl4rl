MECHANISM: Evaluation-only logit sharpening

HYPOTHESIS: Multiplying ensemble logits by 1.02 will preserve exactly 9,133 predictions while lowering validation cross-entropy below 0.246640940094.

INTENDED_EDIT: Apply a positive calibration scale after multi-view logit averaging, without changing training or class rankings.

EVIDENCE: One-float32-ULP EMA movement lost a correct prediction, showing further EMA tuning is accuracy-sensitive; evaluation-only positive scaling preserves argmax while testing whether label smoothing and ten-view averaging leave logits underconfident.

<<<<<<< SEARCH
        return torch.stack(logits, dim=0).sum(dim=0) / (
            2.0 * sum(view_weights)
        )
=======
        return 1.02 * torch.stack(logits, dim=0).sum(dim=0) / (
            2.0 * sum(view_weights)
        )
>>>>>>> REPLACE