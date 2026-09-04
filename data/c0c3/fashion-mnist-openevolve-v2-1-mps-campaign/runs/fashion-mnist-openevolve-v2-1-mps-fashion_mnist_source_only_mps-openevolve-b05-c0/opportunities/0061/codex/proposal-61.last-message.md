MECHANISM: Mild label-smoothing annealing

HYPOTHESIS: Annealing label smoothing from 0.05 to 0.03 will exceed 9,258 correct predictions by preserving early regularization while reducing late target bias.

INTENDED_EDIT: Replace fixed 0.05 label smoothing with a runtime-neutral linear schedule from 0.05 to 0.03.

EVIDENCE: Fixed 0.05 outperformed hard targets and 0.10 smoothing; the runtime-neutral 0.04 attempt timed out without contradictory accuracy evidence, motivating a schedule centered near that low-strength region.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = step / max(total_steps, 1)
    smoothing = 0.05 - 0.02 * progress
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)
>>>>>>> REPLACE