MECHANISM: Linearly annealed label smoothing

HYPOTHESIS: Decaying label smoothing from 0.04 to 0 over training will exceed 9,319 correct predictions by retaining early regularization while allowing late updates to strengthen hard-label decision margins.

INTENDED_EDIT: Replace fixed label smoothing with a linear schedule based on examples-training progress; preserve the verified evaluation temperature and all other training behavior.

EVIDENCE: Evaluation sharpening from temperature 0.95 through 0.80 repeatedly reduced cross-entropy without changing argmax predictions, indicating underconfident logits; fixed 0.04 smoothing suppresses confidence through the final update, motivating a late transition toward hard-label training.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = min(step / max(total_steps, 1), 1.0)
    label_smoothing = 0.04 * (1.0 - progress)
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE