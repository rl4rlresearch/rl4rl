MECHANISM: Warmup-consistent optimizer initialization

HYPOTHESIS: Initializing AdamW at the schedule’s 20% warmup rate will exceed 9,319 correct predictions by preventing the first update from occurring at full peak learning rate before abruptly dropping.

INTENDED_EDIT: Change AdamW’s initial learning rate from 2.0e-3 to 4.0e-4; retain the existing warmup, cosine schedule, architecture, and regularization.

EVIDENCE: The strongest configuration achieved 9,319 correct, while subsequent fine-grained dropout changes regressed. Its schedule specifies a 0.2× starting multiplier, but the optimizer currently performs its first update at the full 2.0e-3 peak, making warmup consistency an untested optimization improvement.

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