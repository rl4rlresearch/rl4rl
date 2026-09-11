MECHANISM: Accelerated residual-adapter optimization

HYPOTHESIS: Training the zero-initialized statistics head and successful diagonal channel gate at 1.5× the backbone learning rate will exceed 9,345 correct predictions by letting these late-starting residual paths specialize within only 1,042 optimizer steps.

INTENDED_EDIT: Place the statistics head and recalibration parameters in a separate AdamW group whose 1.5× learning-rate multiplier is preserved throughout the existing schedule.

EVIDENCE: The statistics bypass and raw-mean diagonal gate produced the best 9,345-correct design, while added representational paths regressed or exceeded the time limit; accelerating only the already-beneficial 2,058 parameters tests optimization underexposure without adding parameters or meaningful computation.

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
=======
    fast_parameters = []
    base_parameters = []
    for name, parameter in model.named_parameters():
        if (
            name.startswith("statistics_head.")
            or name in ("recalibration_scale", "recalibration_bias")
        ):
            fast_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters},
            {
                "params": fast_parameters,
                "lr": PEAK_LR * 0.2 * 1.5,
                "lr_scale": 1.5,
            },
        ],
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = (
            PEAK_LR * multiplier * group.get("lr_scale", 1.0)
        )
>>>>>>> REPLACE