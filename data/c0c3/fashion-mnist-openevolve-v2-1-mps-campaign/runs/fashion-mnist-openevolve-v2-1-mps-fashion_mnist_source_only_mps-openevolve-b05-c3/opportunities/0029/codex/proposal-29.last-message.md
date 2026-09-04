MECHANISM: Short-memory adaptive second moments

HYPOTHESIS: Using AdamW β₂=0.99 will exceed 9,167 correct predictions by adapting its variance estimates more quickly to the validated 39/64 broad-to-cardinal augmentation transition.

INTENDED_EDIT: Retain the best verified architecture, curriculum, EMA, loss, schedule, and inference ensemble while reducing AdamW’s second-moment decay from its default 0.999 to 0.99.

EVIDENCE: The 39/64 curriculum achieved the best completed result at 9,167 correct, and longer cardinal phases improved accuracy monotonically; with only 1,564 optimizer steps, shorter second-moment memory should reduce stale broad-translation influence during terminal inference-aligned training.

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR * 0.2,
        weight_decay=1.5e-4,
    )
=======
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1.5e-4,
    )
>>>>>>> REPLACE