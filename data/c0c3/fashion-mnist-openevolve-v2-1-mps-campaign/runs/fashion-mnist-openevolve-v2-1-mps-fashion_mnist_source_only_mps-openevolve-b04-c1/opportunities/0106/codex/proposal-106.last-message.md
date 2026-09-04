MECHANISM: Uniform tail-window parameter averaging

HYPOTHESIS: Uniformly averaging the final 100 parameter states will exceed 9,328 correct predictions by matching the successful 0.02 EMA’s effective sample size while eliminating its long geometric tail and greater weighting of the noisiest final updates.

INTENDED_EDIT: Replace EMA collection over the second half of training with an exact uniform average of the final 100 optimizer-step parameter states; continue using final normalization buffers.

EVIDENCE: Constant EMA rates of 0.01 and 0.03 reached only 9,322 and 9,325 correct, while 0.02 reached 9,328, indicating that an averaging horizon near 100 states is best; this patch preserves that horizon with a different weighting profile and less averaging overhead.

<<<<<<< SEARCH
    if progress >= 0.50:
        current_state = optimizer.ema_model.state_dict()
        with torch.no_grad():
            if optimizer.ema_state is None:
                optimizer.ema_state = {
                    name: value.detach().clone()
                    for name, value in current_state.items()
                }
            else:
                for name, value in current_state.items():
                    averaged = optimizer.ema_state[name]
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)

            if step >= total_steps:
                for name, value in current_state.items():
                    value.copy_(optimizer.ema_state[name])
=======
    ema_start_step = max(total_steps - 99, 1)
    if step >= ema_start_step:
        current_state = optimizer.ema_model.state_dict()
        averaging_count = step - ema_start_step + 1
        with torch.no_grad():
            if optimizer.ema_state is None:
                optimizer.ema_state = {
                    name: value.detach().clone()
                    for name, value in current_state.items()
                }
            else:
                averaging_rate = 1.0 / averaging_count
                for name, value in current_state.items():
                    averaged = optimizer.ema_state[name]
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), averaging_rate)
                    else:
                        averaged.copy_(value)

            if step >= total_steps:
                for name, value in current_state.items():
                    value.copy_(optimizer.ema_state[name])
>>>>>>> REPLACE