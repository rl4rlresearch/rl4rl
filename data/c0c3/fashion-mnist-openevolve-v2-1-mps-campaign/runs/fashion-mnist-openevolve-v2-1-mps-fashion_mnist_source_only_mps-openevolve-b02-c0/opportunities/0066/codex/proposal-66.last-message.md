MECHANISM: Sparse per-image crop assignment with calibrated ensemble sharpening

HYPOTHESIS: Preserving the verified per-image translation distribution while eliminating dense coordinate-grid indexing will recover 9,252 correct predictions and complete faster; scaling the unchanged ensemble to 1.20 will then lower cross-entropy without changing argmax predictions.

INTENDED_EDIT: Replace coordinate-based crop gathering with disjoint masked assignments from five contiguous crop views, and increase inference calibration from 1.10 to 1.20.

EVIDENCE: Batch-shared crops completed slightly faster but lost 34 correct predictions, showing that per-image crop diversity matters; the verified 1.05-to-1.10 scale increase preserved all 9,252 correct predictions while lowering cross-entropy.

<<<<<<< SEARCH
        return 1.10 * ensemble_log_probabilities
=======
        return 1.20 * ensemble_log_probabilities
>>>>>>> REPLACE

<<<<<<< SEARCH
    offset_draw = torch.randint(0, 13, (batch,), device=images.device)
    offsets_y = (
        1
        + ((offset_draw >= 7) & (offset_draw < 9)).long()
        - ((offset_draw >= 5) & (offset_draw < 7)).long()
    )
    offsets_x = (
        1
        + ((offset_draw >= 11) & (offset_draw < 13)).long()
        - ((offset_draw >= 9) & (offset_draw < 11)).long()
    )
    batch_index = torch.arange(batch, device=images.device)[:, None, None]
    rows = offsets_y[:, None, None] + torch.arange(
        height, device=images.device
    )[None, :, None]
    columns = offsets_x[:, None, None] + torch.arange(
        width, device=images.device
    )[None, None, :]
    images = padded[:, 0][batch_index, rows, columns].unsqueeze(1).contiguous()
=======
    offset_draw = torch.randint(0, 13, (batch,), device=images.device)
    images = padded[..., 1 : height + 1, 1 : width + 1].clone()

    shift_down = (offset_draw >= 5) & (offset_draw < 7)
    images[shift_down] = padded[
        shift_down, :, :height, 1 : width + 1
    ]

    shift_up = (offset_draw >= 7) & (offset_draw < 9)
    images[shift_up] = padded[
        shift_up, :, 2 : height + 2, 1 : width + 1
    ]

    shift_right = (offset_draw >= 9) & (offset_draw < 11)
    images[shift_right] = padded[
        shift_right, :, 1 : height + 1, :width
    ]

    shift_left = (offset_draw >= 11) & (offset_draw < 13)
    images[shift_left] = padded[
        shift_left, :, 1 : height + 1, 2 : width + 2
    ]
>>>>>>> REPLACE