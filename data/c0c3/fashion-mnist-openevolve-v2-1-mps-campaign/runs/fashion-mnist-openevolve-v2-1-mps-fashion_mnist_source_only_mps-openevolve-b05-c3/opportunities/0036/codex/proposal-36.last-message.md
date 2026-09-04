MECHANISM: Phase-responsive EMA for inference-aligned fine-tuning

HYPOTHESIS: Restoring the verified 39/64 transition and increasing EMA responsiveness during the terminal cardinal phase will exceed 9,167 correct predictions by reducing carryover from broadly translated model states while retaining substantial weight averaging.

INTENDED_EDIT: Switch to the best validated augmentation boundary and raise the EMA update rate from 0.005 to 0.0075 only after the training distribution changes to center/cardinal views.

EVIDENCE: The 39/64 curriculum achieved the best completed result at 9,167 correct, while progressively longer inference-aligned phases improved accuracy from 9,141 at 3/4 to 9,155 at 2/3 and 9,163 at 5/8; this motivates weighting terminal aligned states more strongly.

<<<<<<< SEARCH
    if step * 4 < total_steps * 3:
=======
    if step * 64 < total_steps * 39:
>>>>>>> REPLACE

<<<<<<< SEARCH
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)

        if step >= total_steps:
=======
        else:
            ema_rate = (
                0.005 if step * 64 < total_steps * 39 else 0.0075
            )
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, ema_rate)

        if step >= total_steps:
>>>>>>> REPLACE