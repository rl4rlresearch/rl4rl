MECHANISM: Quadratic-recency half-density tail averaging

HYPOTHESIS: Quadratically weighting every other iterate across the final 10% will exceed 9,283 correct predictions by retaining the completed half-density runtime while aligning averaged weights more closely with terminal BatchNorm statistics.

INTENDED_EDIT: Restore proven five-way cardinal translation augmentation and half-density tail averaging, but weight sampled iterates quadratically by recency instead of linearly.

EVIDENCE: Half-density linear averaging completed with 9,283 correct, while dense linear averaging reached 9,285; stronger recency weighting targets the remaining gap without increasing averaging frequency.

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
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
    optimizer.tail_average_count = 0
    optimizer.tail_average_weight = 0.0
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

    tail_distance = total_steps - step - 1
    if (
        step + 1 >= optimizer.tail_average_start
        and tail_distance % 2 == 0
    ):
        optimizer.tail_average_count += 1
        snapshot_weight = float(optimizer.tail_average_count ** 2)
        optimizer.tail_average_weight += snapshot_weight
        average_weight = snapshot_weight / optimizer.tail_average_weight
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