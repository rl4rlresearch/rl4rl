MECHANISM: Classifier-only tail EMA

HYPOTHESIS: Restricting tail EMA to the parameter-dominant classifier will exceed 9,254 correct predictions by retaining weight-averaging benefits while keeping convolution and BatchNorm parameters aligned with final running statistics.

INTENDED_EDIT: Store the classifier parameters as the EMA targets and leave the entire feature extractor at its final trained state.

EVIDENCE: Full-parameter EMA improved the batch-32 result from 9,250 to 9,254, while BatchNorm previously added 40 correct predictions and the classifier contains 207,954 of 239,634 parameters; this motivates concentrating EMA on the dense head without averaging parameters that determine BatchNorm feature statistics.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.5e-3, weight_decay=1e-4
    )
    optimizer._ema_target_parameters = list(model.classifier.parameters())
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
        parameters = optimizer._ema_target_parameters
>>>>>>> REPLACE