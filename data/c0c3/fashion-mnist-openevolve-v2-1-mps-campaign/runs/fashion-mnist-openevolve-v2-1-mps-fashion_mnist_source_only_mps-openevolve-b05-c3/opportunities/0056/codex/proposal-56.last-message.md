MECHANISM: Faster BatchNorm-buffer EMA

HYPOTHESIS: On the verified 19/32 curriculum, increasing only the BatchNorm-buffer EMA rate from 0.005 to 0.01 will exceed 9,172 correct predictions by reducing normalization-statistic lag while retaining the smoothing that outperformed unaveraged final statistics.

INTENDED_EDIT: Restore default AdamW and the best 19/32 transition, keep parameter EMA at 0.005, and use a moderately faster 0.01 EMA for floating BatchNorm buffers.

EVIDENCE: The default-AdamW 19/32 design achieved 9,172 correct with all tensors averaged at 0.005, whereas excluding BatchNorm buffers fell to 9,168; this supports retaining buffer averaging while testing a midpoint toward faster terminal-distribution adaptation.

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
    ema_parameters = [
        tensor for tensor in model.parameters() if tensor.is_floating_point()
    ]
    ema_buffers = [
        tensor for tensor in model.buffers() if tensor.is_floating_point()
    ]
    optimizer._ema_tensors = ema_parameters + ema_buffers
    optimizer._ema_rates = (
        [0.005] * len(ema_parameters) + [0.01] * len(ema_buffers)
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