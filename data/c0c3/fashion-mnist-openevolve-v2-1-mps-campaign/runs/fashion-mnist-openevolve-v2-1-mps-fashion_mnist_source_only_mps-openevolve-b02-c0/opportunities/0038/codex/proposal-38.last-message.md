MECHANISM: BatchNorm-affine-preserving late weight averaging

HYPOTHESIS: Averaging convolutional and linear parameters over the successful final-5% window while retaining final BatchNorm affine parameters will exceed 9,252 correct predictions, or tie with lower cross-entropy, by reducing checkpoint noise without averaging another component coupled to final running statistics.

INTENDED_EDIT: Record non-BatchNorm trainable parameters when constructing the optimizer and restrict the existing late snapshot average to those parameters.

EVIDENCE: Final-5% parameter averaging achieved 9,252 correct, while averaging BatchNorm running statistics reduced accuracy to 9,248 and shortening the averaging window to 2.5% reduced it to 9,247; preserving final BatchNorm affine parameters is a targeted intermediate treatment of the observed normalization mismatch.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=6.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
    batch_norm_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
        for parameter in module.parameters(recurse=False)
    }
    optimizer._late_average_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in batch_norm_parameters
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
        parameters = optimizer._late_average_parameters
>>>>>>> REPLACE