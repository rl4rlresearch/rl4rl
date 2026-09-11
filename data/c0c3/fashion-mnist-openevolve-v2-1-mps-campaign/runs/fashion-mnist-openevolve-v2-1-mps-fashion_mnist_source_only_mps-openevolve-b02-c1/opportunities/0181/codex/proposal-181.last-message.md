MECHANISM: Per-example mild mixup regularization

HYPOTHESIS: Training with Beta(0.2, 0.2) mixup will learn smoother class boundaries and exceed 9,311 correct predictions without materially increasing verification time.

INTENDED_EDIT: Mix each augmented image with a randomly paired batch image and optimize the correspondingly weighted pair of cross-entropy losses.

EVIDENCE: Evaluation-only TTA tuning plateaued at 9,311 correct, while residual architecture changes reduced accuracy and heavier mechanisms timed out; mixup changes learned boundaries using the existing single forward pass and no additional parameters.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    batch = images.shape[0]
    permutation = torch.randperm(batch, device=images.device)
    concentration = images.new_tensor(0.2)
    mix = torch.distributions.Beta(concentration, concentration).sample((batch,))
    mix = torch.maximum(mix, 1.0 - mix)
    mixed_images = (
        mix.view(batch, 1, 1, 1) * images
        + (1.0 - mix).view(batch, 1, 1, 1) * images[permutation]
    )
    logits = model(mixed_images)
    primary_loss = F.cross_entropy(logits, labels, reduction="none")
    paired_loss = F.cross_entropy(
        logits, labels[permutation], reduction="none"
    )
    return (mix * primary_loss + (1.0 - mix) * paired_loss).mean()
>>>>>>> REPLACE