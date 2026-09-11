MECHANISM: Per-example random crop sampling

HYPOTHESIS: Independently sampling the existing uniform full and central crop distributions for every image will exceed 9,267 correct predictions by reducing within-batch augmentation correlation without changing the successful view distribution, architecture, or loss.

INTENDED_EDIT: Replace batch-shared crop offsets with vectorized per-image offsets extracted from the padded 5×5 crop grid.

EVIDENCE: Evaluation-matched triangular crop sampling regressed to 9,247 correct, showing that changing the crop distribution is harmful; this patch preserves the baseline’s uniform marginals while making each update cover more of that proven distribution.

<<<<<<< SEARCH
    offsets = torch.randint(0, 5, (2,))
    offset_y, offset_x = int(offsets[0]), int(offsets[1])
    central_offsets = torch.randint(1, 4, (2,))
    central_y, central_x = int(central_offsets[0]), int(central_offsets[1])
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    view_a = padded[
        :,
        :,
        offset_y : offset_y + 28,
        offset_x : offset_x + 28,
    ]
    view_b = padded[
        :,
        :,
        4 - offset_y : 4 - offset_y + 28,
        4 - offset_x : 4 - offset_x + 28,
    ]
    central_view = padded[
        :,
        :,
        central_y : central_y + 28,
        central_x : central_x + 28,
    ]
=======
    batch_count = images.shape[0]
    offsets = torch.randint(
        0, 5, (batch_count, 2), device=images.device
    )
    offset_y, offset_x = offsets.unbind(dim=1)
    central_offsets = torch.randint(
        1, 4, (batch_count, 2), device=images.device
    )
    central_y, central_x = central_offsets.unbind(dim=1)
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    crop_grid = (
        padded.unfold(2, 28, 1)
        .unfold(3, 28, 1)
        .permute(0, 2, 3, 1, 4, 5)
    )
    batch_indices = torch.arange(batch_count, device=images.device)
    view_a = crop_grid[batch_indices, offset_y, offset_x]
    view_b = crop_grid[
        batch_indices, 4 - offset_y, 4 - offset_x
    ]
    central_view = crop_grid[
        batch_indices, central_y, central_x
    ]
>>>>>>> REPLACE