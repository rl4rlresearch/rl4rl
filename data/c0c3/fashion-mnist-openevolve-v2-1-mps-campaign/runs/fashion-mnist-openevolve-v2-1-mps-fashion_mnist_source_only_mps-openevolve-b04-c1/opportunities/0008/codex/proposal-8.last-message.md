MECHANISM: Paired-view ensemble training

HYPOTHESIS: Training the same two-view logit ensemble used at validation will exceed 9,237 correct predictions by directly optimizing complementary original and mirrored predictions.

INTENDED_EDIT: Evaluate every training image in both orientations, average the paired logits, and compute cross-entropy on the resulting ensemble prediction.

EVIDENCE: Evaluation-time mirror averaging improved the same architecture from 9,194 to 9,212 correct; directly training that successful predictor is an informative way to align optimization with validation behavior while preserving the proven EMA configuration.

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
    batch_size = images.shape[0]
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_logits = model(paired_images)
    logits = 0.5 * (
        paired_logits[:batch_size] + paired_logits[batch_size:]
    )
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
>>>>>>> REPLACE