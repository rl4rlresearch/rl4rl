MECHANISM: Terminal-stat-aligned parameter EMA

HYPOTHESIS: Increasing the parameter-EMA update rate from 0.02 to 0.04 will exceed 9,328 correct predictions by reducing lag between averaged weights and the terminal BatchNorm statistics while retaining meaningful late-step noise suppression.

INTENDED_EDIT: Shorten the second-half parameter-averaging horizon from roughly 50 to 25 optimizer steps, while continuing to copy BatchNorm buffers directly.

EVIDENCE: EMA-averaging BatchNorm buffers scored 9,327 versus the 9,328 baseline, suggesting terminal normalization statistics are preferable; moving averaged parameters closer to those statistics tests the complementary alignment direction without changing training dynamics or runtime.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.04)
>>>>>>> REPLACE