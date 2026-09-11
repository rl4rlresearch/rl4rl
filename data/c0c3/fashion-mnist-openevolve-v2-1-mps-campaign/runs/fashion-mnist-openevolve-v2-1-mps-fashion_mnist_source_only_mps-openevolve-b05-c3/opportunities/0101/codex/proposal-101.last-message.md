MECHANISM: Terminal hard-label fine-tuning

HYPOTHESIS: Disabling label smoothing during the final 13/32 of training will exceed 9,206 correct predictions by preserving early regularization while strengthening true-class margins during the cleaner augmentation and faster-EMA phase.

INTENDED_EDIT: Retain 0.02 label smoothing during broad-shift training, then switch to ordinary cross-entropy when the terminal augmentation phase begins.

EVIDENCE: The qualified model remained underconfident—1.05× evaluation sharpening preserved 9,206 predictions while lowering cross-entropy from 0.235889 to 0.230772—and repeated EMA/calibration refinements did not increase correct count. Terminal hard-label optimization targets learned decision margins while keeping the validated early regularization and evaluation ensemble.

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
    terminal_phase = step * 32 >= total_steps * 19
    smoothing = 0.0 if terminal_phase else 0.02
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)
>>>>>>> REPLACE