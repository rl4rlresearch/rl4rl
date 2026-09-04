MECHANISM: Stronger convolutional-backbone temporal averaging

HYPOTHESIS: Lowering only the four spatial convolution EMA rates from 0.02 to 0.015 will exceed 9,330 correct predictions by smoothing late feature-kernel drift while retaining the proven 0.04 classifier rate and 0.02 rates for normalization and view fusion.

INTENDED_EDIT: Use EMA rate 0.015 for the four spatial convolution weights, 0.04 for classifier parameters, and 0.02 for all remaining learned parameters.

EVIDENCE: Raising the spatial convolution EMA rate to 0.025 reduced correctness from 9,330 to 9,326, while classifier-specific faster averaging remains best; this directly motivates testing stronger averaging in the opposite direction only for backbone kernels.

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
                            ema_rate = 0.015
                        else:
                            ema_rate = 0.02
>>>>>>> REPLACE