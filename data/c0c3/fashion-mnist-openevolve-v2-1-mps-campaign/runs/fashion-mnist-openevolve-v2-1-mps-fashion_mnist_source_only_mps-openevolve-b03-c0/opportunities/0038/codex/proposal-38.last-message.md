MECHANISM: Deterministic horizontal-reflection augmentation

HYPOTHESIS: Horizontally flipping every other training batch will exceed 9,319 correct predictions by teaching label-preserving reflection invariance without increasing the example budget or materially increasing runtime.

INTENDED_EDIT: Apply a horizontal flip to alternating training batches while leaving the model, optimizer, schedule, dropout, and label smoothing unchanged.

EVIDENCE: Translation augmentation targeted positional sensitivity but timed out, while parameter-free architectural invariance changes did not improve accuracy; alternating horizontal flips test a cheaper, label-preserving invariance using only a single tensor operation on half the batches.

<<<<<<< SEARCH
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    return images, labels
=======
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del total_steps
    if step % 2 == 1:
        images = torch.flip(images, dims=(-1,))
    return images, labels
>>>>>>> REPLACE