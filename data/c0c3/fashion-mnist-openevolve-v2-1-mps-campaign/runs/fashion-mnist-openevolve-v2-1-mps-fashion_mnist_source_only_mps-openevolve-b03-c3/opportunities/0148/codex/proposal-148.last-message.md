MECHANISM: Fine-grained cardinal-asymmetry TTA interpolation

HYPOTHESIS: Full-rate pooling with vertical/horizontal radius-one weights of 0.1075/0.0725 will improve validation_score by retaining at least 9,284 correct predictions while lowering cross-entropy relative to the 0.105/0.075 design.

INTENDED_EDIT: Restore the validated full learning rate for both pooling gates and test the midpoint between the 9,284-correct 0.105/0.075 TTA setting and the lower-cross-entropy but 9,283-correct 0.11/0.07 setting.

EVIDENCE: Increasing cardinal contrast monotonically reduced cross-entropy from 0.2081595 at 0.10/0.08 to 0.2081436 at 0.105/0.075 and 0.2081308 at 0.11/0.07, but the last setting lost one correct prediction; meanwhile half-rate pooling regressed from 9,284 to 9,281 correct, so full-rate pooling should be restored before refining the apparent decision-boundary optimum.

<<<<<<< SEARCH
                    if delta_x == 0:
                        weight = 0.10
                    elif delta_y == 0:
                        weight = 0.08
=======
                    if delta_x == 0:
                        weight = 0.1075
                    elif delta_y == 0:
                        weight = 0.0725
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay, no_decay, pooling = [], [], []
    for name, parameter in model.named_parameters():
        if name.endswith("max_logit"):
            pooling.append(parameter)
        elif parameter.ndim > 1:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4, "lr_scale": 1.0},
            {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
            {"params": pooling, "weight_decay": 0.0, "lr_scale": 0.5},
        ],
        lr=2.0e-3,
        betas=(0.9, 0.96),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
        betas=(0.9, 0.96),
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * group["lr_scale"] * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE