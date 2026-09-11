MECHANISM: Final-window Polyak weight averaging

HYPOTHESIS: Averaging the final 64 low-learning-rate parameter states will improve validation_score beyond 9281.414347011598 by reducing endpoint noise while preserving the successful training trajectory.

INTENDED_EDIT: Uniformly average the last 64 post-update model states, install the average after the final step, and use the best-verified evaluation scale of 1.20514.

EVIDENCE: The baseline reached 9,281 correct, while broader, narrower, gated, and augmented variants failed to improve it; final-iterate averaging tests an orthogonal variance-reduction mechanism without changing the verified architecture, loss, or example exposure.

<<<<<<< SEARCH
        if not self.training:
            logits = 1.20485 * logits
=======
        if not self.training:
            logits = 1.20514 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
=======
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    averaging_start = max(1, total_steps - 63)
    if completed >= averaging_start:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        averaged = getattr(optimizer, "_averaged_parameters", None)
        averaged_steps = getattr(optimizer, "_averaged_steps", 0)
        with torch.no_grad():
            if averaged is None:
                averaged = [
                    parameter.detach().clone() for parameter in parameters
                ]
                averaged_steps = 1
            else:
                averaged_steps += 1
                weight = 1.0 / averaged_steps
                for average, parameter in zip(averaged, parameters):
                    average.lerp_(parameter.detach(), weight)
            optimizer._averaged_parameters = averaged
            optimizer._averaged_steps = averaged_steps
            if completed == total_steps:
                for parameter, average in zip(parameters, averaged):
                    parameter.copy_(average)
>>>>>>> REPLACE