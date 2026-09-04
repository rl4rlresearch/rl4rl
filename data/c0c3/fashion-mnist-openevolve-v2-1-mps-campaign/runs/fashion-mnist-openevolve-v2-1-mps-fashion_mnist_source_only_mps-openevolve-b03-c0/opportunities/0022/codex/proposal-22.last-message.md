MECHANISM: Terminal label-smoothing annealing

HYPOTHESIS: Retaining 0.04 smoothing for 90% of training and annealing it to zero over the final 10% will exceed 9,319 correct predictions by preserving its verified generalization benefit while sharpening class margins near convergence.

INTENDED_EDIT: Replace constant label smoothing with a late linear decay from 0.04 to 0.0.

EVIDENCE: Constant 0.04 smoothing achieved 9,316–9,319 correct, while hard labels produced substantially lower cross-entropy but only 9,270 correct; a terminal anneal tests whether their complementary benefits can be combined.

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
    progress = min(step / max(total_steps - 1, 1), 1.0)
    if progress < 0.9:
        smoothing = 0.04
    else:
        smoothing = 0.04 * (1.0 - progress) / 0.1
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)
>>>>>>> REPLACE