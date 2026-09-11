MECHANISM: Paired-flip deep supervision with terminal weight averaging

HYPOTHESIS: Training both horizontal orientations with ensemble-aligned supervision and evaluating an EMA of the final optimization trajectory will exceed the best prior result of 9,276 correct predictions.

INTENDED_EDIT: Replace stochastic flipping with paired-view training, blend per-view and ensemble losses, and maintain a 0.99-decay parameter EMA that is installed after the final optimizer step.

EVIDENCE: Paired-flip deep supervision achieved 9,276 correct, outperforming stochastic flips at 9,247 and translation augmentation at 9,141; EMA directly extends the strongest verified mechanism by reducing late-step parameter noise without changing its augmentation assumptions.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
    optimizer._ema_parameters = []
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    augmented = images.clone()
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    augmented[flip_mask] = torch.flip(augmented[flip_mask], dims=(-1,))
    return augmented, labels
=======
    del step, total_steps
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    del step, total_steps
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_logits = model(paired_images)
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)
    ensemble_logits = 0.5 * (original_logits + flipped_logits)

    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )
    return 0.5 * (ensemble_loss + view_loss)
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> None:
    peak_lr = 3.0e-3
    warmup_steps = max(1, int(0.05 * total_steps))
=======
) -> None:
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    ema_parameters = optimizer._ema_parameters
    with torch.no_grad():
        if not ema_parameters:
            optimizer._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
            ema_parameters = optimizer._ema_parameters
        else:
            for ema_parameter, parameter in zip(ema_parameters, parameters):
                ema_parameter.mul_(0.99).add_(parameter, alpha=0.01)

        if step + 1 == total_steps:
            for parameter, ema_parameter in zip(parameters, ema_parameters):
                parameter.copy_(ema_parameter)

    peak_lr = 3.0e-3
    warmup_steps = max(1, int(0.05 * total_steps))
>>>>>>> REPLACE