MECHANISM: Terminal-aligned BatchNorm statistics with parameter-only EMA

HYPOTHESIS: On the verified 19/32 curriculum, retaining final-phase BatchNorm statistics while averaging only learned parameters will exceed 9,172 correct predictions by eliminating normalization-statistic lag across the augmentation shift.

INTENDED_EDIT: Restore the validated 48-unit head and 19/32 transition, and exclude BatchNorm buffers from EMA so evaluation uses statistics accumulated from the terminal inference-aligned distribution.

EVIDENCE: The 48-unit 19/32 design achieved the best completed result at 9,172 correct; the weaker 52-unit head achieved 9,164, and the curriculum shift makes the current 0.005 EMA of BatchNorm buffers a targeted source of stale broad-augmentation statistics.

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 52),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(52, 10),
=======
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer._ema_tensors = [
        tensor
        for tensor in tuple(model.parameters()) + tuple(model.buffers())
        if tensor.is_floating_point()
    ]
=======
    optimizer._ema_tensors = [
        tensor for tensor in model.parameters() if tensor.is_floating_point()
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step * 64 < total_steps * 39:
=======
    if step * 32 < total_steps * 19:
>>>>>>> REPLACE