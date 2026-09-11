MECHANISM: EMA-free late adaptation for learned detail filters

HYPOTHESIS: Excluding `detail_kernels` from the final EMA interpolation will exceed 9,166 correct predictions by preserving their most recently learned edge and texture adaptations while retaining EMA regularization for the rest of the model.

INTENDED_EDIT: Continue tracking all parameter EMAs, but leave the learned zero-mean detail kernels at their final optimizer values when applying the end-of-training EMA blend.

EVIDENCE: Learning the zero-DC detail kernels improved correctness from 9,162 to 9,166, while additional constraints reduced accuracy; this motivates allowing the successful parameterization to adapt fully without changing its architecture, learning rate, or runtime.

<<<<<<< SEARCH
    optimizer._ema_updates = 0
    return optimizer
=======
    optimizer._ema_updates = 0
    optimizer._ema_no_final_blend = {id(model.detail_kernels)}
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
            for shadow, parameter in zip(optimizer._ema_shadow, parameters):
                parameter.lerp_(shadow / correction, 0.2861344814300537109375)
=======
            for shadow, parameter in zip(optimizer._ema_shadow, parameters):
                if id(parameter) not in optimizer._ema_no_final_blend:
                    parameter.lerp_(
                        shadow / correction,
                        0.2861344814300537109375,
                    )
>>>>>>> REPLACE