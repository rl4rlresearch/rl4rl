MECHANISM: Exponential moving-average checkpoint smoothing

HYPOTHESIS: Evaluating a 0.99-decay moving average of the best model’s training trajectory will exceed 9,273 correct predictions by reducing noise from its 1,564 small-batch updates without altering its successful architecture or learning-rate schedule.

INTENDED_EDIT: Preserve the current model and training procedure, maintain an exponential moving average of every learned parameter after optimizer steps, and install those averaged parameters after the final step for validation.

EVIDENCE: The current pairwise-refinement model is best at 9,273 correct, while residual refinement and a 10× lower terminal learning rate fell to 9,268 and 9,270; this motivates smoothing the verified trajectory rather than changing its features or endpoint schedule again.

<<<<<<< SEARCH
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.5e-4,
        betas=(0.9, 0.99),
    )
=======
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.5e-4,
        betas=(0.9, 0.99),
    )
    optimizer.ema_model = model
    optimizer.ema_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
=======
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    ema_decay = 0.99
    model = optimizer.ema_model
    ema_parameters = optimizer.ema_parameters
    with torch.no_grad():
        for averaged, parameter in zip(ema_parameters, model.parameters()):
            averaged.lerp_(parameter, 1.0 - ema_decay)
        if completed == max(total_steps, 1):
            for parameter, averaged in zip(model.parameters(), ema_parameters):
                parameter.copy_(averaged)
>>>>>>> REPLACE