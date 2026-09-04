MECHANISM: Temperature-sharpened dual-statistic channel attention

HYPOTHESIS: Sharpening the gate response by 25% will exceed 9,162 correct predictions by helping the zero-initialized attention learn stronger channel distinctions within 1,564 optimizer steps while retaining its bounded scale range.

INTENDED_EDIT: Multiply the fused mean–maximum gate signal by 1.25 before `tanh`, preserving identity initialization, parameter count, descriptors, and runtime.

EVIDENCE: Dual-statistic attention improved correctness from 9,153 to 9,162, while changing its equal descriptor mixture reduced accuracy; strengthening the successful gate response is therefore a cleaner next test than altering descriptor fusion again.

<<<<<<< SEARCH
        channel_scale = 1.0 + 0.5 * torch.tanh(channel_signal)
=======
        channel_scale = 1.0 + 0.5 * torch.tanh(1.25 * channel_signal)
>>>>>>> REPLACE