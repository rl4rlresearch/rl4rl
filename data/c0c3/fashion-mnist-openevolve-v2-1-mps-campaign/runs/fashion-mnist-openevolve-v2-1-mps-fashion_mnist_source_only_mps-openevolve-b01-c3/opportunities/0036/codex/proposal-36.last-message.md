MECHANISM: Hard-target tail refinement with recency-weighted averaging

HYPOTHESIS: Annealing label smoothing to zero before the proven final-10% linear tail average will exceed 9,285 correct predictions by retaining early regularization while sharpening terminal decision boundaries.

INTENDED_EDIT: Restore cardinal translation training and linear recency-weighted averaging, while holding 0.02 label smoothing through 75% of training and reducing it to zero by the averaging window.

EVIDENCE: Final-10% linear averaging achieved the best verified result at 9,285 correct, outperforming uniform averaging and the shorter 5% window; this motivates refining the terminal trajectory without shortening its successful averaging span.

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
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")

    row_offsets = torch.randint(0, 5, (batch, 1), device=images.device)
    rows = row_offsets + torch.arange(height, device=images.device).unsqueeze(0)
    row_index = rows[:, None, :, None].expand(
        batch, channels, height, padded.shape[-1]
    )
    images = padded.gather(2, row_index)

    col_offsets = torch.randint(0, 5, (batch, 1), device=images.device)
    cols = col_offsets + torch.arange(width, device=images.device).unsqueeze(0)
=======
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
    row_offsets = torch.tensor(
        (1, 0, 2, 1, 1), device=images.device
    )[positions].unsqueeze(1)
    col_offsets = torch.tensor(
        (1, 1, 1, 0, 2), device=images.device
    )[positions].unsqueeze(1)

    rows = row_offsets + torch.arange(height, device=images.device).unsqueeze(0)
    row_index = rows[:, None, :, None].expand(
        batch, channels, height, padded.shape[-1]
    )
    images = padded.gather(2, row_index)

    cols = col_offsets + torch.arange(width, device=images.device).unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
) -> torch.Tensor:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    smoothing = 0.02 * max(
        0.0,
        min(1.0, (0.90 - progress) / 0.15),
    )
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier

    if step + 1 >= optimizer.tail_average_start:
        optimizer.tail_average_count += 1
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
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