MECHANISM: Sparse tail weight averaging on reliable maximum attention

HYPOTHESIS: Restoring plain paired-view cross-entropy and averaging sparse low-learning-rate tail checkpoints will exceed 9,322 correct predictions by recovering the proven 9,320-correct objective while reducing late-training parameter noise.

INTENDED_EDIT: Remove the harmful ensemble-aware loss and install a running average sampled every 20 optimizer steps during the final 20% of training, copying it into the model after the last step.

EVIDENCE: Plain paired-view cross-entropy with maximum attention achieved 9,320 correct, while the ensemble-aware objective reduced that architecture to 9,307. Top-four attention added only two correct predictions and repeatedly approached or exceeded the runtime limit, motivating a low-overhead optimization refinement of the reliable maximum-attention design.

<<<<<<< SEARCH
    paired_logits = model(paired_images)
    individual_loss = F.cross_entropy(paired_logits, paired_labels)

    batch_size = images.shape[0]
    ensemble_log_probabilities = torch.logsumexp(
        torch.stack(
            (
                F.log_softmax(paired_logits[:batch_size], dim=1),
                F.log_softmax(paired_logits[batch_size:], dim=1),
            )
        ),
        dim=0,
    ) - math.log(2.0)
    ensemble_loss = F.nll_loss(ensemble_log_probabilities, labels)
    return 0.5 * (individual_loss + ensemble_loss)
=======
    return F.cross_entropy(model(paired_images), paired_labels)
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier

    calls = getattr(optimizer, "_tail_average_calls", 0) + 1
    optimizer._tail_average_calls = calls
    averaging_start = max(1, math.ceil(0.8 * total_steps))
    final_step = calls >= total_steps
    take_snapshot = (
        calls >= averaging_start
        and ((calls - averaging_start) % 20 == 0 or final_step)
    )
    if not take_snapshot:
        return

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    with torch.no_grad():
        averages = getattr(optimizer, "_tail_averages", None)
        if averages is None:
            averages = [parameter.detach().clone() for parameter in parameters]
            optimizer._tail_averages = averages
            optimizer._tail_average_count = 1
        else:
            count = optimizer._tail_average_count + 1
            for average, parameter in zip(averages, parameters):
                average.lerp_(parameter, 1.0 / count)
            optimizer._tail_average_count = count

        if final_step:
            for parameter, average in zip(parameters, averages):
                parameter.copy_(average)
>>>>>>> REPLACE