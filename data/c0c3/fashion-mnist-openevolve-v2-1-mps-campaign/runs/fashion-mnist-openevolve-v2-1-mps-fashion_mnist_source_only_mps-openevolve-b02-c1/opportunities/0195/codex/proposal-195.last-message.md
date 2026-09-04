MECHANISM: Short-horizon Adam second-moment adaptation

HYPOTHESIS: Reducing AdamW’s second-moment decay from 0.999 to 0.99 will exceed 9,318 correct predictions by adapting its preconditioner more quickly during the 2,084-step warmup-and-cosine schedule.

INTENDED_EDIT: Set AdamW betas to `(0.9, 0.99)` while preserving the model, loss, augmentation, learning-rate schedule, and weight averaging.

EVIDENCE: Label-smoothing, pooling, and TTA variants failed to improve beyond 9,318 correct; with only 2,084 optimizer steps, the default 0.999 second-moment timescale spans much of training, motivating a distinct optimization-level change.

<<<<<<< SEARCH
    return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
=======
    return torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        betas=(0.9, 0.99),
        weight_decay=2e-4,
    )
>>>>>>> REPLACE