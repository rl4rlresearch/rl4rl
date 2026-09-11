MECHANISM: Short linear learning-rate warmup with cosine decay

HYPOTHESIS: A 5% warmup from 20% to full learning rate followed by cosine decay will exceed 9,260 correct predictions by stabilizing early AdamW updates without materially reducing the fixed-budget learning-rate integral.

INTENDED_EDIT: Initialize AdamW at 5e-4 and replace immediate cosine decay with a 5%-warmup, 95%-cosine schedule; preserve the verified architecture, loss, EMA, BatchNorm-buffer mixture, and evaluation calibration.

EVIDENCE: The 9,260-correct baseline uses immediate cosine decay, while label-smoothing, augmentation, dropout, residual, and capacity changes all reduced correctness. The only schedule experiment used a substantially more aggressive 20% peak hold and timed out, leaving a conservative compute-neutral warmup unresolved.

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.5e-3, weight_decay=1e-4
    )
=======
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5.0e-4, weight_decay=1e-4
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    progress = step / max(total_steps, 1)
    warmup_fraction = 0.05
    if progress < warmup_fraction:
        multiplier = 0.2 + 0.8 * progress / warmup_fraction
    else:
        decay_progress = (progress - warmup_fraction) / (
            1.0 - warmup_fraction
        )
        multiplier = 0.5 * (
            1.0 + math.cos(math.pi * min(decay_progress, 1.0))
        )
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE