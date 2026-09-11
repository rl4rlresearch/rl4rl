MECHANISM: Foreach-accelerated short-horizon full-state EMA

HYPOTHESIS: A computationally batched 0.985-decay full-state EMA will finish within the time limit and exceed 9,334 correct predictions by tracking late low-learning-rate convergence more closely than the successful 0.99 EMA.

INTENDED_EDIT: Replace dictionary-based sparse EMA with batched foreach updates over all floating parameters and BatchNorm buffers, while shortening the EMA horizon from 0.99 to 0.985.

EVIDENCE: Sparse full-state 0.99 EMA improved the same model from 9,322 to 9,334 correct, whereas the proposed 0.985 horizon produced no accuracy evidence because both prior implementations timed out; batching the state updates directly tests that localized hypothesis with less Python overhead.

<<<<<<< SEARCH
    optimizer._averaging_model = model
    optimizer._averaged_state = {}
    optimizer._averaging_last_step = None
=======
    optimizer._averaging_values = tuple(
        value
        for value in model.state_dict().values()
        if torch.is_floating_point(value)
    )
    optimizer._averaged_values = None
    optimizer._averaging_last_step = None
>>>>>>> REPLACE

<<<<<<< SEARCH
    if should_average:
        model = optimizer._averaging_model
        averaged_state = optimizer._averaged_state
        last_step = optimizer._averaging_last_step
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if not torch.is_floating_point(value):
                    continue
                if name not in averaged_state:
                    averaged_state[name] = value.detach().clone()
                else:
                    elapsed = max(step - last_step, 1)
                    decay = 0.99 ** elapsed
                    averaged_state[name].mul_(decay).add_(
                        value.detach(),
                        alpha=1.0 - decay,
                    )
        optimizer._averaging_last_step = step

        if step >= total_steps:
            with torch.no_grad():
                for name, value in model.state_dict().items():
                    if name in averaged_state:
                        value.copy_(averaged_state[name])
=======
    if should_average:
        values = optimizer._averaging_values
        averaged_values = optimizer._averaged_values
        last_step = optimizer._averaging_last_step
        with torch.no_grad():
            if averaged_values is None:
                averaged_values = [
                    value.detach().clone() for value in values
                ]
            else:
                elapsed = max(step - last_step, 1)
                decay = 0.985 ** elapsed
                torch._foreach_mul_(averaged_values, decay)
                torch._foreach_add_(
                    averaged_values,
                    values,
                    alpha=1.0 - decay,
                )
        optimizer._averaged_values = averaged_values
        optimizer._averaging_last_step = step

        if step >= total_steps:
            with torch.no_grad():
                torch._foreach_copy_(values, averaged_values)
>>>>>>> REPLACE