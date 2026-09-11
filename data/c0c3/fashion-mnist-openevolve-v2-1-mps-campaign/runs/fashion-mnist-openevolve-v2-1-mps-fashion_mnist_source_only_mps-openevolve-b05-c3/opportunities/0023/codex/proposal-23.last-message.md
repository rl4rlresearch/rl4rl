MECHANISM: Phase-aware EMA acceleration

HYPOTHESIS: Doubling the EMA update rate during the inference-aligned cardinal phase will exceed 9,167 correct predictions by reducing stale broad-translation influence on final weights and BatchNorm statistics.

INTENDED_EDIT: Restore the best 39/64 augmentation boundary and change EMA interpolation from 0.005 to 0.01 only after that boundary.

EVIDENCE: Extending the cardinal phase from one-quarter through 25/64 monotonically raised validation correct from 9,141 to 9,167; faster late EMA tracking tests whether emphasizing that proven terminal distribution yields another gain.

<<<<<<< SEARCH
    if step * 3 < total_steps * 2:
=======
    if step * 64 < total_steps * 39:
>>>>>>> REPLACE

<<<<<<< SEARCH
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)
=======
        else:
            ema_update_rate = (
                0.005 if step * 64 < total_steps * 39 else 0.01
            )
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, ema_update_rate)
>>>>>>> REPLACE