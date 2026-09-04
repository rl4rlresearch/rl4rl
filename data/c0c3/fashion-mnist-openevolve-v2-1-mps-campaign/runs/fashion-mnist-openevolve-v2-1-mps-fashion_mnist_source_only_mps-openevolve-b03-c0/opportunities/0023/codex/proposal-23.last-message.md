MECHANISM: Early hard-target smoothing ramp

HYPOTHESIS: Ramping label smoothing from 0 to 0.04 during the first 10% of training will exceed 9,319 correct predictions by strengthening early class-learning gradients while retaining smoothing throughout convergence.

INTENDED_EDIT: Linearly warm label smoothing from 0 to 0.04 over the first 10% of optimizer steps, then hold it at 0.04.

EVIDENCE: Constant 0.04 smoothing reached 9,319 correct, while annealing it to zero late fell to 9,309; this indicates smoothing is especially valuable near convergence, motivating the reverse schedule.

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
    smoothing_progress = min(step / max(0.10 * total_steps, 1.0), 1.0)
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=0.04 * smoothing_progress,
    )
>>>>>>> REPLACE