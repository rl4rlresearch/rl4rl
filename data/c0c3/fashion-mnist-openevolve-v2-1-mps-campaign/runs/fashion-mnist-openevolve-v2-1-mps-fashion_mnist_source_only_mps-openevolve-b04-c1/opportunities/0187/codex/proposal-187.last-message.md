MECHANISM: Faster convolution-only EMA alignment

HYPOTHESIS: Raising only convolutional-backbone EMA rates from 0.02 to 0.025 will exceed 9,330 correct predictions by better aligning feature kernels with final BatchNorm statistics without disturbing the proven classifier or normalization averaging.

INTENDED_EDIT: Use EMA rate 0.025 for the four spatial convolution weights, retain 0.04 for classifier parameters, and retain 0.02 for view fusion and BatchNorm affine parameters.

EVIDENCE: EMA-averaging BatchNorm buffers fell to 9,327 and using final BatchNorm affine parameters fell to 9,328, while classifier-specific faster EMA reached 9,330; this motivates a smaller, isolated acceleration of upstream convolution weights.

<<<<<<< SEARCH
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
=======
                        if name.startswith("classifier."):
                            ema_rate = 0.04
                        elif name in {
                            "stem.0.weight",
                            "residual1.0.weight",
                            "transition.0.weight",
                            "residual2.0.weight",
                        }:
                            ema_rate = 0.025
                        else:
                            ema_rate = 0.02
>>>>>>> REPLACE