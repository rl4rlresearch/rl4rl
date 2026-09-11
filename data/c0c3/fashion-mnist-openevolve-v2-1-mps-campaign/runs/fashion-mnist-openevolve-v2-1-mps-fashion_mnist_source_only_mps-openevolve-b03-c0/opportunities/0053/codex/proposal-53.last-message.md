MECHANISM: Warmup-consistent optimizer initialization

HYPOTHESIS: Starting AdamW at the schedule’s 0.2× warmup rate instead of taking one anomalous full-rate update will exceed 9,319 correct predictions by stabilizing early feature learning.

INTENDED_EDIT: Initialize the optimizer at 4.0e-4; retain the existing warmup, cosine schedule, loss, architecture, and verified 0.80 evaluation temperature.

EVIDENCE: Temperature calibration repeatedly preserved exactly 9,319 correct predictions, so further gains require changing training. The current optimizer takes its first step at 2.0e-3 before the scheduler reduces it to approximately 4.0e-4, contradicting the intended warmup.

<<<<<<< SEARCH
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-3,
        betas=(0.9, 0.99),
        weight_decay=1.0e-3,
    )
=======
    return torch.optim.AdamW(
        model.parameters(),
        lr=4.0e-4,
        betas=(0.9, 0.99),
        weight_decay=1.0e-3,
    )
>>>>>>> REPLACE