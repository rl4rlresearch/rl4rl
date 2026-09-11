MECHANISM: Curriculum-synchronized label-smoothing decay

HYPOTHESIS: Linearly decaying label smoothing from 0.02 to zero during the inference-aligned terminal phase will exceed 9,193 correct predictions by preserving early regularization while sharpening late decision boundaries.

INTENDED_EDIT: Retain the verified residual model and 19/32 augmentation curriculum, but anneal label smoothing only after the curriculum transition with negligible computational overhead.

EVIDENCE: The residual architecture with the 19/32 curriculum achieved the best completed result of 9,193 correct, while several compute-adding extensions timed out; this isolates a lightweight optimization change at the already validated transition.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    if step * 32 < total_steps * 19:
        label_smoothing = 0.02
    else:
        terminal_progress = (
            step * 32 - total_steps * 19
        ) / max(total_steps * 13, 1)
        label_smoothing = 0.02 * max(0.0, 1.0 - terminal_progress)
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE