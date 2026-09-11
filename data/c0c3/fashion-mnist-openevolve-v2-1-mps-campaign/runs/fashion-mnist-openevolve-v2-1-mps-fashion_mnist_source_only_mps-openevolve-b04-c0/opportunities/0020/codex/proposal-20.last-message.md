MECHANISM: Midpoint coherent EMA horizon refinement

HYPOTHESIS: An EMA decay of 0.955 will exceed the 0.96 design’s 9,358 correct predictions—or tie it with lower cross-entropy—by balancing late-trajectory responsiveness against variance reduction.

INTENDED_EDIT: Increase parameter and floating-buffer EMA update rates from 4% to 4.5%, preserving all other behavior.

EVIDENCE: Decay 0.96 achieved 9,358 correct with 0.193622 cross-entropy, while 0.95 tied the correct count but worsened cross-entropy to 0.193787; testing their midpoint directly refines the newly bracketed optimum.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.04)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.04)
=======
                    average.lerp_(parameter.detach(), 0.045)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.045)
>>>>>>> REPLACE