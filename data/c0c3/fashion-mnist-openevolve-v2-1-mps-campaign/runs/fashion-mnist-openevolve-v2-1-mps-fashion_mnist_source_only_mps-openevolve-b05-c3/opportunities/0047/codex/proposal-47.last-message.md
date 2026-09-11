MECHANISM: Short-memory adaptive variance for curriculum shifts

HYPOTHESIS: Using AdamW β₂=0.99 with the verified 39/64 augmentation transition will exceed 9,167 correct predictions by letting adaptive learning rates respond to terminal inference-aligned gradients instead of retaining second-moment statistics from most of the broad-translation phase.

INTENDED_EDIT: Restore the best verified 39/64 augmentation boundary and shorten AdamW’s second-moment memory from the default 0.999 to 0.99.

EVIDENCE: The 39/64 curriculum produced the best completed result at 9,167 correct; its deliberate mid-run distribution change motivates testing an optimizer whose variance estimate adapts substantially faster than the default, whose approximately 693-step half-life spans much of this 1,564-step run.

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

<<<<<<< SEARCH
    if step * 3 < total_steps * 2:
=======
    if step * 64 < total_steps * 39:
>>>>>>> REPLACE