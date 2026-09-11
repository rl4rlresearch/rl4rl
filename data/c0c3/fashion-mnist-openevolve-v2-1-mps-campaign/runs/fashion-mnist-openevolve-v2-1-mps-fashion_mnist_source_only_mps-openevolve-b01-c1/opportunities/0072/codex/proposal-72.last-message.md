MECHANISM: Per-example translation sampling

HYPOTHESIS: Sampling crop offsets independently per image will exceed 9,247 correct predictions by increasing coverage of the proven translation distribution and preventing batch-correlated augmentation without introducing new distortions.

INTENDED_EDIT: Replace batch-wide full and central crop offsets with per-example offsets selected from an unfolded crop bank; preserve opposite-offset pairing, losses, architecture, and evaluation.

EVIDENCE: Rotation and cutout regressed to 9,162 and 9,223 correct, while architectural changes also underperformed; this motivates improving diversity within the successful ±2-pixel crop geometry rather than adding invariances or capacity.

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
    batch_size = images.shape[0]
    batch_indices = torch.arange(batch_size, device=images.device)
    offsets = torch.randint(
        0, 5, (batch_size, 2), device=images.device
    )
    offset_y, offset_x = offsets.unbind(dim=1)
    central_offsets = torch.randint(
        1, 4, (batch_size, 2), device=images.device
    )
    central_y, central_x = central_offsets.unbind(dim=1)
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    crop_bank = (
        padded.unfold(2, 28, 1)
        .unfold(3, 28, 1)
        .permute(0, 2, 3, 1, 4, 5)
    )
    view_a = crop_bank[batch_indices, offset_y, offset_x]
    view_b = crop_bank[
        batch_indices,
        4 - offset_y,
        4 - offset_x,
    ]
    central_view = crop_bank[
        batch_indices,
        central_y,
        central_x,
    ]
>>>>>>> REPLACE