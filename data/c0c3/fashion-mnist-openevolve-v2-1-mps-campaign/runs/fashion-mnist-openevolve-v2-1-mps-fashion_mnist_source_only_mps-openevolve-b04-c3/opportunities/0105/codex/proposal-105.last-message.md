MECHANISM: Per-example integer translation augmentation

HYPOTHESIS: Adding independent uniform ±1-pixel translations alongside horizontal flips will increase validation_correct above 9,243 after 100,000 examples without changing the 245,040-parameter model.

INTENDED_EDIT: Generate one random 1-pixel crop from a replicate-padded image for every training example, then apply the existing random horizontal flip.

EVIDENCE: Repeated temperature and power-mean refinements preserved exactly 9,243 predictions and have reached a cross-entropy plateau near 0.208833005, motivating a training-time invariance change capable of improving class rankings.

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
        images.flip(-1),
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
    del step, total_steps
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    crops = F.unfold(padded, kernel_size=images.shape[-2:])
    crop_indices = torch.randint(
        crops.shape[-1],
        (images.shape[0],),
        device=images.device,
    )
    images = crops.gather(
        2,
        crop_indices[:, None, None].expand(-1, crops.shape[1], 1),
    ).squeeze(2).reshape_as(images)
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    return images, labels
>>>>>>> REPLACE