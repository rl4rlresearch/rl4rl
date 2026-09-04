MECHANISM: Late-stage dropout annealing

HYPOTHESIS: Annealing classifier dropout from 0.1 to zero during the final 30% of training will exceed 9,345 correct predictions by preserving early regularization while allowing cleaner convergence within the fixed 1,042-update budget.

INTENDED_EDIT: Keep dropout at 0.1 for the first 70% of updates, then linearly reduce it to zero without changing parameters or computational cost.

EVIDENCE: Raw-mean gating produced the best 9,345-correct result, while many added representation and loss variants timed out; adapting the existing regularizer directly targets limited optimization exposure without adding runtime-sensitive computation.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = (step + 1) / max(total_steps, 1)
    if progress <= 0.7:
        dropout_probability = 0.1
    else:
        dropout_probability = 0.1 * (1.0 - progress) / 0.3
    model.classifier[4].p = dropout_probability
    return F.cross_entropy(model(images), labels)
>>>>>>> REPLACE