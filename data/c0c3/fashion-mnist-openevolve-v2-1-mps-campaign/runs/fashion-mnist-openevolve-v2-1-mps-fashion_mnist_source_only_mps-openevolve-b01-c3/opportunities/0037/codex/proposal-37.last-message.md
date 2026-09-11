MECHANISM: Terminal BatchNorm affine anchoring during tail averaging

HYPOTHESIS: Leaving BatchNorm scale and bias at their terminal values while linearly averaging all other parameters will exceed 9,285 correct predictions by better matching the terminal running statistics.

INTENDED_EDIT: Preserve the proven final-10% recency-weighted average, but exclude BatchNorm trainable parameters from it.

EVIDENCE: Linear recency weighting improved uniform final-10% averaging from 9,282 to 9,285 correct, supporting reduced mismatch with terminal BatchNorm state; this targets that mismatch without extra passes or averaging work.

<<<<<<< SEARCH
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return optimizer
=======
    batch_norm_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
        for parameter in module.parameters()
    }
    optimizer.tail_average_model_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in batch_norm_parameter_ids
    ]
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in optimizer.tail_average_model_parameters
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
=======
        parameters = optimizer.tail_average_model_parameters
>>>>>>> REPLACE