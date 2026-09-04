MECHANISM: Upper-side coherent EMA horizon refinement

HYPOTHESIS: An EMA decay of 0.9625 will exceed the 0.96 design’s 9,358 correct predictions—or tie it with lower cross-entropy—by adding modest variance reduction without returning to the less responsive 0.97 horizon.

INTENDED_EDIT: Reduce parameter and floating-buffer EMA update rates from 4% to 3.75%, preserving all other training behavior.

EVIDENCE: Decay 0.96 achieved the best result at 9,358 correct and 0.193622 cross-entropy; 0.97 fell to 9,356 correct, while the lower-side midpoint 0.955 fell to 9,357, making 0.9625 the most informative untested local bracket.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.04)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.04)
=======
                    average.lerp_(parameter.detach(), 0.0375)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.0375)
>>>>>>> REPLACE