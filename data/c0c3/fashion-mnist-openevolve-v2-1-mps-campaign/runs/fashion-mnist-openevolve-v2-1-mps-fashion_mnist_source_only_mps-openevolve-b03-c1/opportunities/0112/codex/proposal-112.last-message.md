MECHANISM: Moderated detail-filter adaptation rate

HYPOTHESIS: Training `detail_kernels` at 0.70× the base learning rate will exceed 9,166 correct predictions by balancing useful task-specific adaptation against the degradation observed at 2× learning rate.

INTENDED_EDIT: Place `detail_kernels` in a separate AdamW parameter group with a persistent 0.70× learning-rate multiplier, leaving weight decay, scheduling, EMA, architecture, and runtime otherwise unchanged.

EVIDENCE: Learning the kernels improved correctness from 9,162 to 9,166, but doubling their learning rate reduced it to 9,149; an intermediate slower rate tests whether the current kernels over-adapt slightly while retaining the benefit lost when they are fixed.

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
    named_parameters = list(model.named_parameters())
    detail = [
        parameter
        for name, parameter in named_parameters
        if name == "detail_kernels"
    ]
    decay = [
        parameter
        for name, parameter in named_parameters
        if parameter.ndim > 1 and name != "detail_kernels"
    ]
    no_decay = [
        parameter
        for _, parameter in named_parameters
        if parameter.ndim <= 1
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": detail,
                "weight_decay": 4e-4,
                "lr": 0.70 * 1.25e-4,
                "lr_scale": 0.70,
            },
            {"params": decay, "weight_decay": 4e-4, "lr_scale": 1.0},
            {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
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
        group["lr"] = 1.25e-3 * multiplier * group["lr_scale"]
>>>>>>> REPLACE