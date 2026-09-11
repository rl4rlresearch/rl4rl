MECHANISM: Terminal hard-label boundary refinement

HYPOTHESIS: Removing label smoothing only during the final 13/32 training phase will exceed 9,206 correct predictions by refining class boundaries under the validation-aligned ±1-shift curriculum without adding runtime or parameters.

INTENDED_EDIT: Retain 0.02 label smoothing during broad augmentation, then switch to ordinary cross-entropy when the existing terminal augmentation and fast EMA phase begins.

EVIDENCE: Per-view and post-ensemble calibration preserved exactly 9,206 correct, while clean-view weighting reduced correctness to 9,202; this motivates changing learned decision boundaries rather than further manipulating unchanged TTA predictions.

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
    label_smoothing = 0.02 if step * 32 < total_steps * 19 else 0.0
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE