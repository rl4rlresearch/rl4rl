MECHANISM: Normalized mild focal reweighting

HYPOTHESIS: Reweighting training examples toward low-confidence classifications while retaining half-weight on easy examples will exceed 9,360 correct predictions by improving decision boundaries without changing the verified architecture or evaluation ensemble.

INTENDED_EDIT: Replace uniform cross-entropy with detached confidence-based weights normalized to preserve the average gradient scale.

EVIDENCE: Architecture changes fell to 9,326–9,337 correct, EMA-consistent BatchNorm fell to 9,351, and probability-space TTA fell to 9,359; this motivates preserving the 9,360-correct model and changing only how its fixed training exposure prioritizes ambiguous examples.

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
    del step, total_steps
    logits = model(images)
    per_example_loss = F.cross_entropy(
        logits, labels, reduction="none"
    )
    with torch.no_grad():
        true_probability = torch.exp(-per_example_loss)
        difficulty = (1.0 - true_probability).clamp_min(0.0).sqrt()
        weights = 0.5 + 0.5 * difficulty
        weights = weights / weights.mean()
    return (weights * per_example_loss).mean()
>>>>>>> REPLACE