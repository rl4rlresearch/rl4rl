MECHANISM: EMA-consistent BatchNorm statistics

HYPOTHESIS: Averaging BatchNorm running means and variances alongside the learned parameters will exceed 9,328 correct predictions by evaluating a normalization state aligned with the successful parameter-EMA trajectory.

INTENDED_EDIT: Extend the existing 0.02 EMA to every floating-point model-state tensor while continuing to copy integer BatchNorm counters directly.

EVIDENCE: EMA-rate changes to 0.01 and 0.03 reduced validation correct to 9,322 and 9,325, so the 0.02 parameter horizon should be preserved; the current implementation pairs those averaged parameters with final-step normalization buffers, making state alignment a distinct runtime-neutral target.

<<<<<<< SEARCH
    optimizer.ema_parameter_names = {
        name for name, _ in model.named_parameters()
    }
=======
    optimizer.ema_parameter_names = {
        name
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }
>>>>>>> REPLACE