MECHANISM: Time-budgeted intermediate-batch optimization

HYPOTHESIS: Batch size 48 with a 1.7e-3 peak learning rate will exceed 9,290 correct predictions by gaining 33% more optimizer updates than batch size 64 while avoiding the timeout observed at batch size 32.

INTENDED_EDIT: Reduce the batch size from 64 to 48 and scale both AdamW’s initial learning rate and the scheduled peak learning rate to 1.7e-3.

EVIDENCE: Moving from batch size 128 to 64 improved correctness from 9,257 to 9,290, while batch size 32 with 1.4e-3 failed to finish; batch size 48 directly tests the remaining feasible optimization-granularity range.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 2.0
=======
BATCH_SIZE = 48
GRAD_CLIP_NORM = 2.0
>>>>>>> REPLACE

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
        lr=1.7e-3,
        betas=(0.9, 0.99),
        weight_decay=1.0e-3,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 1.7e-3 * multiplier
>>>>>>> REPLACE