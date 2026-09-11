MECHANISM: Final-state BatchNorm affine/statistic alignment

HYPOTHESIS: Keeping BatchNorm affine parameters at their final values while EMA-averaging convolutional and classifier parameters will exceed 9,330 correct predictions by aligning normalization scales and offsets with the final BatchNorm running statistics.

INTENDED_EDIT: Exclude all BatchNorm weight and bias parameters from EMA so they are copied directly each step, matching the existing treatment of BatchNorm buffers.

EVIDENCE: EMA-averaging BatchNorm running statistics reduced correctness from 9,330 to 9,327, indicating final normalization statistics are preferable; retaining final BatchNorm affine parameters tests the complementary way to eliminate the current EMA-weight/final-statistic mismatch.

<<<<<<< SEARCH
    optimizer.ema_parameter_names = {
        name for name, _ in model.named_parameters()
    }
=======
    batch_norm_prefixes = (
        "stem.1.",
        "residual1.1.",
        "transition.1.",
        "residual2.1.",
    )
    optimizer.ema_parameter_names = {
        name
        for name, _ in model.named_parameters()
        if not name.startswith(batch_norm_prefixes)
    }
>>>>>>> REPLACE