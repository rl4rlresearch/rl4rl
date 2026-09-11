MECHANISM: Selective matrix-weight regularization

HYPOTHESIS: Applying meaningful decoupled weight decay only to convolutional and linear weight tensors will exceed 9,280 correct predictions by strengthening the regularization that prior dropout evidence indicates is beneficial, without penalizing normalization scales or biases.

INTENDED_EDIT: Replace uniform 2e-4 AdamW decay with 1e-2 decay for parameters having at least two dimensions and zero decay for biases and normalization parameters.

EVIDENCE: Reducing classifier dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, indicating insufficient regularization hurts; selective AdamW decay tests complementary regularization with negligible runtime impact.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay_parameters = []
    no_decay_parameters = []
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": 1e-2},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=8e-4,
    )
>>>>>>> REPLACE