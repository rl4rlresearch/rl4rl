MECHANISM: Exposure-normalized tail weight averaging

HYPOTHESIS: Reducing the EMA update rate from 0.01 to 0.0025 will exceed 9,250 correct predictions by restoring the same examples-based averaging horizon that benefited the batch-128 model.

INTENDED_EDIT: Change only the EMA interpolation rate to account for the fourfold increase in optimizer steps caused by reducing batch size from 128 to 32.

EVIDENCE: Tail EMA improved the batch-128 design from 9,168 to 9,170 correct; at batch 32, the unchanged per-step decay averages only one quarter as many examples, motivating a fourfold lower interpolation rate.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.01)
=======
                    average.lerp_(parameter.detach(), 0.0025)
>>>>>>> REPLACE