MECHANISM: Antithetic translation-and-flip ensemble supervision

HYPOTHESIS: Jointly supervising complementary translated views and their mirrors, including loss on their four-logit average, will exceed 9,124 correct predictions by extending the successful flip-aligned training to the translation ensemble used at evaluation.

INTENDED_EDIT: Keep training batches unmodified, then construct two complementary random translations and both horizontal orientations inside the loss; blend per-view cross-entropy with four-view ensemble cross-entropy.

EVIDENCE: Paired-view ensemble-aligned supervision raised validation correctness from 9,063 to 9,124, while evaluation averages translated and mirrored views; this directly tests whether aligning the remaining translation dimension yields another accuracy gain.

<<<<<<< SEARCH
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    offsets = torch.randint(0, 5, (2,))
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    offset_y, offset_x = int(offsets[0]), int(offsets[1])
    images = padded[:, :, offset_y : offset_y + 28, offset_x : offset_x + 28]
    flip_mask = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
=======
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    paired_logits = model(torch.cat((images, images.flip(-1)), dim=0))
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)
    individual_loss = F.cross_entropy(
        paired_logits,
        labels.repeat(2),
        label_smoothing=0.02,
    )
    ensemble_loss = F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=0.02,
    )
    return 0.5 * individual_loss + 0.5 * ensemble_loss
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    offsets = torch.randint(0, 5, (2,))
    offset_y, offset_x = int(offsets[0]), int(offsets[1])
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    view_a = padded[
        :,
        :,
        offset_y : offset_y + 28,
        offset_x : offset_x + 28,
    ]
    view_b = padded[
        :,
        :,
        4 - offset_y : 4 - offset_y + 28,
        4 - offset_x : 4 - offset_x + 28,
    ]
    logits = model(
        torch.cat(
            (view_a, view_b, view_a.flip(-1), view_b.flip(-1)),
            dim=0,
        )
    )
    logits_a, logits_b, flipped_a, flipped_b = logits.chunk(4, dim=0)
    individual_loss = F.cross_entropy(
        logits,
        labels.repeat(4),
        label_smoothing=0.02,
    )
    ensemble_loss = F.cross_entropy(
        0.25 * (logits_a + logits_b + flipped_a + flipped_b),
        labels,
        label_smoothing=0.02,
    )
    return 0.5 * individual_loss + 0.5 * ensemble_loss
>>>>>>> REPLACE