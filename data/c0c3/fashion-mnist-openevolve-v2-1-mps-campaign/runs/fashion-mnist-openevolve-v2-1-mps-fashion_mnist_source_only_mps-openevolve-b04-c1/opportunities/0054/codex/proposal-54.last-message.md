MECHANISM: EMA-aligned dropout annealing

HYPOTHESIS: Holding dropout at 0.15 for the first half of training and annealing it to zero during EMA collection will exceed 9,316 correct predictions by reducing late gradient noise without weakening the learning-rate schedule.

INTENDED_EDIT: Keep the successful architecture and early regularization unchanged, then linearly remove classifier dropout over the final half of the fixed exposure budget.

EVIDENCE: Lowering the terminal learning-rate floor reduced validation correct from 9,316 to 9,294, suggesting late learning should remain active; annealing dropout instead preserves update magnitude while making late optimization and EMA snapshots less stochastic.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    logits = model(images)
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = min(step / max(total_steps, 1), 1.0)
    dropout_decay = max(progress - 0.50, 0.0) / 0.50
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
>>>>>>> REPLACE