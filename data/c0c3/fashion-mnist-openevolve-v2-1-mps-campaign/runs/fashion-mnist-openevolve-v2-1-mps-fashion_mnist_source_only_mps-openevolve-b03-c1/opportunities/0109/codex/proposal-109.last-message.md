MECHANISM: Two-rate detail-filter optimization

HYPOTHESIS: Training the successful learned zero-mean detail kernels at twice the base learning rate will exceed 9,166 correct predictions by accelerating their task-specific adaptation during the fixed two-pass exposure without changing model capacity or runtime materially.

INTENDED_EDIT: Isolate `detail_kernels` in their own AdamW parameter group with a 2× learning-rate multiplier while preserving weight decay, EMA, scheduling, and all other behavior.

EVIDENCE: Learning the detail kernels improved correctness from 9,162 to 9,166, showing that their adaptation is useful; this tests whether the 27 filter parameters are currently adapting too slowly under the optimizer rate shared with the much larger network.

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
    detail = [model.detail_kernels]
    decay = [
        parameter
        for parameter in model.parameters()
        if parameter.ndim > 1 and parameter is not model.detail_kernels
    ]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": detail,
                "weight_decay": 4e-4,
                "lr": 2.5e-4,
                "lr_scale": 2.0,
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
        group["lr"] = 1.25e-3 * group.get("lr_scale", 1.0) * multiplier
>>>>>>> REPLACE