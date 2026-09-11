MECHANISM: Fused linear-recency tail averaging

HYPOTHESIS: Restoring the 9,285-correct cardinal-shift design while fusing its final-10% parameter averaging will recover at least 9,285 correct predictions and finish within the verification time limit.

INTENDED_EDIT: Restore five-way cardinal translation augmentation and exact final-10% linear recency averaging, using foreach tensor operations to reduce averaging overhead.

EVIDENCE: Reference Design 3 achieved the best verified result of 9,285 correct in 59.96 seconds; later implementations repeatedly timed out, making a computationally fused reproduction of that proven method the most informative next test.

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
    optimizer.tail_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in optimizer.tail_parameters
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
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
        with torch.no_grad():
            torch._foreach_lerp_(
                optimizer.tail_average_parameters,
                optimizer.tail_parameters,
                average_weight,
            )
            if step + 1 == total_steps:
                torch._foreach_copy_(
                    optimizer.tail_parameters,
                    optimizer.tail_average_parameters,
                )
>>>>>>> REPLACE