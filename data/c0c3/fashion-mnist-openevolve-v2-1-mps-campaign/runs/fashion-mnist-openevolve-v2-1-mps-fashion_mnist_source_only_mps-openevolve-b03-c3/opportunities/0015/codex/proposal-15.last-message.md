MECHANISM: Pure flip-ensemble supervision

HYPOTHESIS: Removing the harmful EMA and increasing flip-averaged loss weight from 0.75 to 1.0 will exceed 9,233 correct predictions by fully aligning optimization with the flip-averaged validation decision rule.

INTENDED_EDIT: Restore Reference Design 3’s ordinary AdamW trajectory and replace its 25% individual-view/75% ensemble loss with pure cross-entropy on averaged flip-pair logits.

EVIDENCE: Reference Design 3 achieved 9,233 correct after increasing ensemble-loss weight from 0.5 to 0.75, while the 0.99 EMA reduced performance to 9,191; this motivates removing EMA and testing the remaining supervision-weight endpoint.

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
    )
    optimizer._ema_parameters = None
    optimizer._ema_buffers = [
        buffer for buffer in model.buffers() if buffer.is_floating_point()
    ]
    optimizer._ema_buffer_values = None
    optimizer._ema_updates = 0
    optimizer._ema_total_steps = total_steps
    return optimizer
=======
    del total_steps
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)

    individual_loss = F.cross_entropy(
        paired_logits, paired_labels, label_smoothing=0.03
    )
    ensemble_loss = F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=0.03,
    )
    return 0.5 * individual_loss + 0.5 * ensemble_loss
=======
    paired_logits = model(paired_images)
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)

    return F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=0.03,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer._ema_updates += 1

    with torch.no_grad():
        if optimizer._ema_parameters is None:
            optimizer._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
            optimizer._ema_buffer_values = [
                buffer.detach().clone() for buffer in optimizer._ema_buffers
            ]
        else:
            for average, parameter in zip(
                optimizer._ema_parameters, parameters
            ):
                average.lerp_(parameter.detach(), 0.01)
            for average, buffer in zip(
                optimizer._ema_buffer_values, optimizer._ema_buffers
            ):
                average.lerp_(buffer.detach(), 0.01)

        if optimizer._ema_updates >= optimizer._ema_total_steps:
            for parameter, average in zip(
                parameters, optimizer._ema_parameters
            ):
                parameter.copy_(average)
            for buffer, average in zip(
                optimizer._ema_buffers, optimizer._ema_buffer_values
            ):
                buffer.copy_(average)
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE