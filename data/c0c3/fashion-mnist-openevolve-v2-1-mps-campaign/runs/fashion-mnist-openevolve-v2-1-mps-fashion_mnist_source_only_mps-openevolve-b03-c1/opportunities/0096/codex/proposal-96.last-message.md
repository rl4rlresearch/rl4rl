MECHANISM: Selective learning-rate acceleration for adaptive detail filters

HYPOTHESIS: Training the successful zero-DC 3×3 detail kernels at 1.5× the base learning rate will exceed 9,166 correct predictions by increasing their adaptation within the fixed exposure budget while preserving the verified architecture and initialization.

INTENDED_EDIT: Place `detail_kernels` in a dedicated AdamW parameter group with a 1.5× learning-rate multiplier; retain all other optimization, regularization, model, and evaluation behavior.

EVIDENCE: Learning the original unconstrained zero-DC kernels improved correctness from 9,162 to 9,166, whereas expanding them to 5×5 fell to 9,148 and constraining their energy fell to 9,153; this motivates faster optimization of the established useful parameterization rather than adding capacity or constraints.

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
    detail = [
        parameter
        for name, parameter in model.named_parameters()
        if name == "detail_kernels"
    ]
    decay = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.ndim > 1 and name != "detail_kernels"
    ]
    no_decay = [
        parameter
        for parameter in model.parameters()
        if parameter.ndim <= 1
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": detail,
                "weight_decay": 4e-4,
                "lr": 1.875e-4,
                "lr_scale": 1.5,
            },
            {"params": decay, "weight_decay": 4e-4},
            {"params": no_decay, "weight_decay": 0.0},
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
        group["lr"] = (
            1.25e-3 * multiplier * group.get("lr_scale", 1.0)
        )
>>>>>>> REPLACE