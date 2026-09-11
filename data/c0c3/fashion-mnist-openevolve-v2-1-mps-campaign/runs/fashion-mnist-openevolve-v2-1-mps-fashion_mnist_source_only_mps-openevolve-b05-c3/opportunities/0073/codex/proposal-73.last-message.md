MECHANISM: Curriculum-boundary EMA restart

HYPOTHESIS: Resetting the all-tensor EMA at the 19/32 augmentation transition will exceed 9,193 correct predictions by removing stale broad-shift averaging while preserving tail smoothing of parameters and BatchNorm statistics.

INTENDED_EDIT: Reinitialize every EMA tensor from the live model exactly when cardinal augmentation begins, then continue the validated 0.005 EMA through the terminal phase.

EVIDENCE: The residual 19/32 design with all-tensor EMA achieved the best completed result of 9,193 correct, while parameter-only EMA fell to 9,168; restarting rather than narrowing the EMA preserves the beneficial normalization-statistic averaging while aligning it with the validated terminal curriculum.

<<<<<<< SEARCH
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)
=======
        elif (
            (step - 1) * 32 < total_steps * 19
            and step * 32 >= total_steps * 19
        ):
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.copy_(tensor)
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)
>>>>>>> REPLACE