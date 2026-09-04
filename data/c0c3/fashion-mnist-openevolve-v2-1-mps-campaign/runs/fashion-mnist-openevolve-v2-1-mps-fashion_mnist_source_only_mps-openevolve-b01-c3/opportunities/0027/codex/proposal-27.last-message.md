MECHANISM: Extended-window tail weight averaging

HYPOTHESIS: Uniformly averaging the final 15% of iterates will exceed 9,282 correct predictions because the 10% window outperformed the more terminal-proximal 5% window, indicating that broader low-learning-rate temporal coverage improves decision accuracy.

INTENDED_EDIT: Restore evaluation-matched cardinal translation training and add online parameter averaging over the final 15% of optimizer steps while preserving the proven architecture, schedule, and inference ensemble.

EVIDENCE: Final-10% averaging achieved 9,282 correct versus 9,275 for final-5% averaging on the same 249,961-parameter design; extending the window is the direct next test of the observed accuracy benefit from broader averaging.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=2e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5.0e-4,
        weight_decay=2e-4,
    )
    optimizer.tail_average_start = max(1, int(0.85 * total_steps))
    optimizer.tail_average_count = 0
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    batch, channels, height, width = images.shape
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")

    row_offsets = torch.randint(0, 5, (batch, 1), device=images.device)
    rows = row_offsets + torch.arange(height, device=images.device).unsqueeze(0)
=======
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
    row_offsets = torch.tensor(
        (1, 0, 2, 1, 1), device=images.device
    )[positions].unsqueeze(1)
    col_offsets = torch.tensor(
        (1, 1, 1, 0, 2), device=images.device
    )[positions].unsqueeze(1)

    rows = row_offsets + torch.arange(height, device=images.device).unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    col_offsets = torch.randint(0, 5, (batch, 1), device=images.device)
    cols = col_offsets + torch.arange(width, device=images.device).unsqueeze(0)
=======
    cols = col_offsets + torch.arange(width, device=images.device).unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier

    if step + 1 >= optimizer.tail_average_start:
        optimizer.tail_average_count += 1
        average_weight = 1.0 / optimizer.tail_average_count
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            for average, parameter in zip(
                optimizer.tail_average_parameters,
                parameters,
            ):
                average.lerp_(parameter, average_weight)

            if step + 1 == total_steps:
                for parameter, average in zip(
                    parameters,
                    optimizer.tail_average_parameters,
                ):
                    parameter.copy_(average)
>>>>>>> REPLACE