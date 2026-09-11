MECHANISM: Cyclic integer-translation augmentation

HYPOTHESIS: Adding uniformly cycled two-pixel translations to the reliable hard-maximum model will exceed 9,322 correct predictions by improving spatial robustness without the runtime risk of ranking-based attention.

INTENDED_EDIT: Extend the existing paired-flip training augmentation with a deterministic 5×5 translation cycle using inexpensive replicated padding and cropping.

EVIDENCE: Hard-maximum attention reliably reached 9,320 correct, while most attention variants regressed or timed out and top-four improved by only two predictions; the paired loss already supplies both horizontal orientations, making spatial translation an informative orthogonal augmentation.

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
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )

    padding = 2
    translation_width = 2 * padding + 1
    phase = step % (translation_width * translation_width)
    offset_y = phase // translation_width
    offset_x = phase % translation_width
    padded_images = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    images = padded_images[
        :,
        :,
        offset_y : offset_y + images.shape[-2],
        offset_x : offset_x + images.shape[-1],
    ].contiguous()
    return images, labels
>>>>>>> REPLACE