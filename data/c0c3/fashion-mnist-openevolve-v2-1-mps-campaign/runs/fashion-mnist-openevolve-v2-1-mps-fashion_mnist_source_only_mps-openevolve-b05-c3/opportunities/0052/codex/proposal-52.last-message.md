MECHANISM: Curriculum-responsive BatchNorm EMA

HYPOTHESIS: Using a four-times-faster EMA for BatchNorm statistics on the best 19/32 curriculum will exceed 9,172 correct predictions by aligning normalization with the terminal inference-matched distribution while retaining strong weight averaging.

INTENDED_EDIT: Restore default AdamW β₂ and the verified 19/32 transition, then update floating BatchNorm buffers at EMA rate 0.02 while keeping learned parameters at 0.005.

EVIDENCE: The default-AdamW 19/32 design achieved the best completed result at 9,172 correct, whereas β₂=0.99 achieved 9,163; the terminal distribution shift specifically motivates faster adaptation of normalization statistics.

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1.5e-4,
    )
    optimizer._ema_tensors = [
        tensor
        for tensor in tuple(model.parameters()) + tuple(model.buffers())
        if tensor.is_floating_point()
    ]
=======
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR * 0.2,
        weight_decay=1.5e-4,
    )
    ema_parameters = tuple(
        tensor for tensor in model.parameters() if tensor.is_floating_point()
    )
    ema_buffers = tuple(
        tensor for tensor in model.buffers() if tensor.is_floating_point()
    )
    optimizer._ema_tensors = ema_parameters + ema_buffers
    optimizer._ema_rates = (
        (0.005,) * len(ema_parameters) + (0.02,) * len(ema_buffers)
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step * 64 < total_steps * 39:
=======
    if step * 32 < total_steps * 19:
>>>>>>> REPLACE

<<<<<<< SEARCH
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)
=======
            for average, tensor, rate in zip(
                optimizer._ema_values,
                optimizer._ema_tensors,
                optimizer._ema_rates,
            ):
                average.lerp_(tensor, rate)
>>>>>>> REPLACE