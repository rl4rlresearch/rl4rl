MECHANISM: Lag-compensated BatchNorm buffer averaging

HYPOTHESIS: Keeping parameter EMA at the best-tested 0.96 decay while increasing floating-buffer updates to 0.0625 will exceed 9,358 correct predictions—or tie with lower cross-entropy—by compensating for BatchNorm’s intrinsic running-statistic lag.

INTENDED_EDIT: Preserve the 4% parameter EMA update and increase only the floating-buffer EMA update from 4% to 6.25%.

EVIDENCE: Extending parameter EMA to BatchNorm buffers previously improved validation_correct from 9,323 to 9,335, while joint EMA-decay refinement established 0.96 as the best parameter horizon; decoupling the buffer rate now targets BatchNorm’s additional momentum-induced lag without disturbing that optimum.

<<<<<<< SEARCH
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.04)
=======
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.0625)
>>>>>>> REPLACE