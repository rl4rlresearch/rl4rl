MECHANISM: Paired-view translation augmentation

HYPOTHESIS: Replacing the algebraically redundant pre-pair horizontal flip with random one-pixel translations will exceed 9,282 correct predictions by teaching translation robustness while preserving the proven mirrored-pair objective.

INTENDED_EDIT: Sample the center or one of four cardinal one-pixel translations independently per training image using replicated-edge padding; paired mirror training, architecture, optimizer, and EMA remain unchanged.

EVIDENCE: Paired-view training reached 9,282 correct, but its existing random pre-flip only swaps the two paired views and adds no augmentation. Translation ensembling regressed to 9,187, indicating shifted predictions are not yet reliable; training on those shifts directly tests whether learned invariance can recover their potential without increasing forward-pass count.

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
    del step, total_steps
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    translated = torch.stack(
        (
            padded[:, :, 1:29, 1:29],
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        ),
        dim=1,
    )
    choices = torch.randint(
        translated.shape[1],
        (images.shape[0],),
        device=images.device,
    )
    images = translated[
        torch.arange(images.shape[0], device=images.device),
        choices,
    ]
    return images, labels
>>>>>>> REPLACE