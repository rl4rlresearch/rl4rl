MECHANISM: Update-equivalent learning-rate scaling

HYPOTHESIS: Scaling the batch-48 learning-rate schedule by 1.2× will exceed 9,280 correct predictions by approximating the cumulative optimization of batch size 40 without its additional steps or runtime.

INTENDED_EDIT: Increase the initial, peak, and terminal learning rates by 20% while preserving batch size, warmup, cosine schedule, architecture, augmentation, and evaluation.

EVIDENCE: Moving from batch size 64 to 48 increased optimizer steps by 33% and improved validation_correct from 9,238 to 9,280, while batch size 40 timed out; a 1.2× learning-rate scale matches the attempted 48/40 increase in update opportunity at unchanged computational cost.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=9.6e-4, weight_decay=2e-4)
>>>>>>> REPLACE

<<<<<<< SEARCH
    if progress < warmup:
        learning_rate = 8e-4 + (3.0e-3 - 8e-4) * progress / warmup
    else:
        decay = (progress - warmup) / (1.0 - warmup)
        multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * decay))
        learning_rate = 3.0e-3 * multiplier
=======
    if progress < warmup:
        learning_rate = 9.6e-4 + (3.6e-3 - 9.6e-4) * progress / warmup
    else:
        decay = (progress - warmup) / (1.0 - warmup)
        multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * decay))
        learning_rate = 3.6e-3 * multiplier
>>>>>>> REPLACE