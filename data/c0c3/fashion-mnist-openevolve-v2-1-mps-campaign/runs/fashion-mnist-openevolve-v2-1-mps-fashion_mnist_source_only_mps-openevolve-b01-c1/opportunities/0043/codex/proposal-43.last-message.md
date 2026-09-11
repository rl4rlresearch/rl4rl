MECHANISM: Per-example Monte Carlo crop sampling

HYPOTHESIS: Sampling crop offsets independently for every image will exceed 9,239 correct predictions by reducing batch-correlated augmentation noise and exposing each optimizer step to broader translation coverage while preserving the proven architecture, six-view objective, and cosine curriculum.

INTENDED_EDIT: Replace the three batch-shared crop offsets with independently sampled per-example offsets, using an unfolded padded-image view for vectorized crop selection.

EVIDENCE: The cosine-curriculum design remains best at 9,239 correct, while subsequent loss-schedule, capacity, and feature-head changes regressed; this motivates a compute-neutral improvement to the remaining batch-correlated augmentation mechanism without disturbing the successful model or objective.

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
    batch_indices = torch.arange(images.size(0), device=images.device)
    offsets = torch.randint(
        0, 5, (images.size(0), 2), device=images.device
    )
    central_offsets = torch.randint(
        1, 4, (images.size(0), 2), device=images.device
    )
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    crops = (
        padded.unfold(2, 28, 1)
        .unfold(3, 28, 1)
        .permute(0, 2, 3, 1, 4, 5)
    )
    view_a = crops[
        batch_indices,
        offsets[:, 0],
        offsets[:, 1],
    ]
    view_b = crops[
        batch_indices,
        4 - offsets[:, 0],
        4 - offsets[:, 1],
    ]
    central_view = crops[
        batch_indices,
        central_offsets[:, 0],
        central_offsets[:, 1],
    ]
>>>>>>> REPLACE