MECHANISM: Compute-efficient batchwise translation augmentation

HYPOTHESIS: Deterministic one-pixel translations on half the training batches will exceed 9,330 correct predictions by adding spatial robustness without the runtime cost that prevented the prior per-example translation experiment from completing.

INTENDED_EDIT: Replace the prediction-redundant random horizontal flip with evenly cycled up/down/left/right translations implemented by a single batchwise roll and replicated boundary repair.

EVIDENCE: The earlier one-pixel translation design timed out rather than producing negative accuracy evidence, while horizontal flipping is already made exactly invariant by view fusion; this patch tests the same unresolved augmentation idea with substantially less indexing and padding overhead.

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
    if step % 2 == 1:
        shifts = ((1, 0), (-1, 0), (0, 1), (0, -1))
        dy, dx = shifts[(step // 2) % len(shifts)]
        images = torch.roll(images, shifts=(dy, dx), dims=(-2, -1))
        if dy == 1:
            images[:, :, 0, :] = images[:, :, 1, :]
        elif dy == -1:
            images[:, :, -1, :] = images[:, :, -2, :]
        if dx == 1:
            images[:, :, :, 0] = images[:, :, :, 1]
        elif dx == -1:
            images[:, :, :, -1] = images[:, :, :, -2]
    return images, labels
>>>>>>> REPLACE