MECHANISM: Conservative adaptive-filter learning rate

HYPOTHESIS: Training the successful zero-DC detail filters at half the backbone learning rate will exceed 9,166 correct predictions by preserving their useful derivative priors while still permitting task-specific adaptation.

INTENDED_EDIT: Place `detail_kernels` in a dedicated AdamW parameter group with a 0.5 learning-rate multiplier; leave all other optimization, architecture, and evaluation behavior unchanged.

EVIDENCE: Making the fixed detail filters trainable improved correctness from 9,162 to 9,166, while expanding the handcrafted representation regressed to 9,114; this directly tests whether gentler adaptation can retain more of the beneficial prior without adding computation or parameters.

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
    detail_parameter = model.detail_kernels
    decay = [
        parameter
        for parameter in model.parameters()
        if parameter.ndim > 1 and parameter is not detail_parameter
    ]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [detail_parameter],
                "weight_decay": 4e-4,
                "lr": 6.25e-5,
                "lr_scale": 0.5,
            },
            {
                "params": decay,
                "weight_decay": 4e-4,
                "lr": 1.25e-4,
                "lr_scale": 1.0,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "lr": 1.25e-4,
                "lr_scale": 1.0,
            },
        ],
        betas=(0.9, 0.99),
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 1.25e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 1.25e-3 * multiplier * group["lr_scale"]
>>>>>>> REPLACE