MECHANISM: Cyclic batch-shared one-pixel translation augmentation

HYPOTHESIS: Cheap deterministic translations on the reliable hard-maximum architecture will exceed 9,322 correct predictions by improving positional robustness without the timeout overhead of per-example augmentation or top-k attention.

INTENDED_EDIT: Replace the redundant preparatory random flip with a replicate-padded crop cycling through all nine one-pixel offsets; paired-view supervision still supplies both horizontal orientations.

EVIDENCE: The hard-maximum model reliably completed with 9,320 correct, while the per-example translation experiment on top-four attention timed out; batch-shared slicing isolates the promising orthogonal augmentation at substantially lower overhead.

<<<<<<< SEARCH
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels
=======
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del total_steps
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    top = step % 3
    left = (step // 3) % 3
    images = padded[:, :, top : top + 28, left : left + 28]
    return images, labels
>>>>>>> REPLACE