MECHANISM: Accelerated adaptive detail-filter learning

HYPOTHESIS: Training the zero-DC detail kernels at 1.5× the backbone learning rate will exceed 9,166 correct predictions by allowing their useful task-specific adaptation to converge further within the fixed 1,564-step budget.

INTENDED_EDIT: Give `detail_kernels` a dedicated AdamW parameter group with a persistent 1.5× learning-rate multiplier while leaving the architecture, regularization, EMA, and backbone schedule unchanged.

EVIDENCE: Making the detail filters trainable improved correctness from 9,162 to 9,166, while fixed-energy normalization regressed to 9,153; this motivates faster unconstrained adaptation rather than another restriction on kernel scale.

<<<<<<< SEARCH
    decay = [parameter for parameter in model.parameters() if parameter.ndim > 1]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 4e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1.25e-4,
        betas=(0.9, 0.99),
    )
=======
    detail_parameters = [model.detail_kernels]
    decay = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.ndim > 1 and name != "detail_kernels"
    ]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 4e-4, "lr_scale": 1.0},
            {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
            {
                "params": detail_parameters,
                "weight_decay": 4e-4,
                "lr": 1.875e-4,
                "lr_scale": 1.5,
            },
        ],
        lr=1.25e-4,
        betas=(0.9, 0.99),
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 1.25e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 1.25e-3 * group["lr_scale"] * multiplier
>>>>>>> REPLACE