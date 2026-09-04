MECHANISM: Short-horizon exponential weight averaging

HYPOTHESIS: Restoring beta2=0.96 and evaluating a decay-0.95 EMA of the final training trajectory will exceed 9,283 correct predictions by suppressing late boundary-crossing parameter noise.

INTENDED_EDIT: Restore the best verified AdamW beta2 and maintain an optimizer-updated parameter EMA that is substituted when the model enters evaluation.

EVIDENCE: Beta2=0.96 achieved the best 9,283 correct, while changing peak rate or terminal floor consistently lost accuracy; this motivates smoothing late updates without altering the proven schedule.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        height, width = images.shape[-2:]
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        ema_parameters = getattr(self, "_ema_parameters", None)
        if ema_parameters is not None:
            with torch.no_grad():
                for parameter, averaged_parameter in zip(
                    self.parameters(), ema_parameters
                ):
                    parameter.copy_(averaged_parameter)

        height, width = images.shape[-2:]
>>>>>>> REPLACE

<<<<<<< SEARCH
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
        betas=(0.9, 0.97),
    )
=======
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
        betas=(0.9, 0.96),
    )
    model._ema_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    optimizer._ema_model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    model = optimizer._ema_model
    with torch.no_grad():
        for index, (averaged_parameter, parameter) in enumerate(
            zip(model._ema_parameters, model.parameters())
        ):
            if averaged_parameter.device != parameter.device:
                averaged_parameter = parameter.detach().clone()
                model._ema_parameters[index] = averaged_parameter
            averaged_parameter.lerp_(parameter.detach(), 0.05)
>>>>>>> REPLACE