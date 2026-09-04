MECHANISM: Terminal label-smoothing annealing

HYPOTHESIS: Linearly reducing label smoothing from 0.02 to zero during the final 13/32 training phase will exceed 9,206 correct predictions by strengthening true-class margins while retaining early regularization.

INTENDED_EDIT: Keep 0.02 label smoothing during broad translation training, then anneal it to zero alongside the cleaner terminal augmentation phase.

EVIDENCE: The verified 1.05× sharpening retained 9,206 correct while reducing cross-entropy from 0.235889 to 0.230772, indicating underconfidence; the prior abrupt hard-label trial timed out and provided no contrary performance evidence.

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
        smoothing = 0.02
    else:
        terminal_progress = (
            step * 32 - total_steps * 19
        ) / max(total_steps * 13, 1)
        smoothing = 0.02 * max(0.0, 1.0 - terminal_progress)
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)
>>>>>>> REPLACE