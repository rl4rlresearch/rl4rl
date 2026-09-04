MECHANISM: Parameter-only EMA with fresh BatchNorm statistics

HYPOTHESIS: Averaging learned parameters while retaining the latest BatchNorm running statistics will exceed 9,282 correct predictions by avoiding normalization-buffer lag in the small-batch EMA model.

INTENDED_EDIT: Track parameter names and apply EMA only to learned parameters; copy BatchNorm buffers directly from the latest training state.

EVIDENCE: EMA at rate 0.01 improved validation correct from 9,215 to 9,237, while the longer 0.005 horizon regressed to 9,233, suggesting excessive state lag is harmful; paired training subsequently raised the same EMA design to 9,282.

<<<<<<< SEARCH
    optimizer.ema_model = model
    optimizer.ema_state = None
    return optimizer
=======
    optimizer.ema_model = model
    optimizer.ema_state = None
    optimizer.ema_parameter_names = {
        name for name, _ in model.named_parameters()
    }
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
                for name, value in current_state.items():
                    averaged = optimizer.ema_state[name]
                    if torch.is_floating_point(averaged):
                        averaged.lerp_(value.detach(), 0.01)
                    else:
                        averaged.copy_(value)
=======
                for name, value in current_state.items():
                    averaged = optimizer.ema_state[name]
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.01)
                    else:
                        averaged.copy_(value)
>>>>>>> REPLACE