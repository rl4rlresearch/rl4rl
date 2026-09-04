MECHANISM: Curriculum-synchronized faster all-tensor EMA

HYPOTHESIS: Increasing the EMA update rate from 0.005 to 0.01 only during the cardinal-augmentation phase will exceed 9,193 correct predictions by reducing broad-phase parameter and BatchNorm lag while retaining late-training smoothing.

INTENDED_EDIT: Preserve the verified residual architecture, 19/32 curriculum, and all-tensor EMA, but shorten the EMA averaging horizon after the curriculum transition.

EVIDENCE: The current all-tensor residual design achieved the best completed result of 9,193 correct; this coefficient-only change targets the stale-state concern behind the unverified EMA-restart experiment without adding meaningful computation.

<<<<<<< SEARCH
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)

        if step >= total_steps:
=======
        else:
            ema_rate = 0.01 if step * 32 >= total_steps * 19 else 0.005
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, ema_rate)

        if step >= total_steps:
>>>>>>> REPLACE