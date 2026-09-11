MECHANISM: Paired-flip ensemble training

HYPOTHESIS: Training the same two-view averaged logits used during evaluation will exceed 9,302 correct predictions by directly optimizing the flip ensemble that already improved accuracy by 12 images.

INTENDED_EDIT: Replace stochastic single-view flips with paired original/reflected views and compute cross-entropy on their averaged logits.

EVIDENCE: Flip test-time ensembling improved validation_correct from 9,290 to 9,302 and reduced cross-entropy from 0.20879 to 0.20463, showing that the paired-view predictor is stronger than either evaluation orientation alone.

<<<<<<< SEARCH
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
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
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_logits = model(paired_images)
    logits, flipped_logits = paired_logits.chunk(2, dim=0)
    ensemble_logits = 0.5 * (logits + flipped_logits)
    return F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE