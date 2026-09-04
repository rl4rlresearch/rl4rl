MECHANISM: State-consistent EMA for BatchNorm statistics

HYPOTHESIS: Averaging floating-point BatchNorm buffers with the model parameters will exceed 9,328 correct predictions by reducing the mismatch between EMA weights and final-model normalization statistics.

INTENDED_EDIT: Extend EMA averaging from learned parameters to every floating-point model state, while continuing to copy integer tracking buffers directly.

EVIDENCE: EMA-aligned dropout and label-smoothing annealing improved validation correct from 9,316 to 9,328; the current implementation averages parameters but pairs them with BatchNorm running statistics copied from the non-EMA model.

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