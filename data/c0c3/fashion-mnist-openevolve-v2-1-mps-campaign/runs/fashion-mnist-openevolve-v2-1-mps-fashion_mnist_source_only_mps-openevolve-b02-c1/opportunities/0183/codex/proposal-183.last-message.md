MECHANISM: Tail-phase dropout annealing

HYPOTHESIS: Linearly removing head dropout during the final 20% of training will consolidate low-learning-rate features and tail-averaged weights, exceeding 9,311 correct predictions without added computation or parameters.

INTENDED_EDIT: Keep dropout at 0.15 for the first 80% of training, then anneal it to zero over the same interval used for parameter averaging.

EVIDENCE: Evaluation-only tuning plateaued at 9,311 correct, while margin training and augmentation-distribution changes reduced correctness; this motivates a low-cost optimization change that preserves the successful model and early regularization while improving final convergence.

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
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    if progress >= 0.8:
        model.classifier[4].p = 0.15 * (1.0 - progress) / 0.2
    return F.cross_entropy(model(images), labels)
>>>>>>> REPLACE