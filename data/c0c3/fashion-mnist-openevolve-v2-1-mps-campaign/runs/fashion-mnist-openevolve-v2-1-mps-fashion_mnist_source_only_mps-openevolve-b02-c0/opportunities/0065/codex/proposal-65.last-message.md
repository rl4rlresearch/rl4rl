MECHANISM: Stratified batch-shared translation

HYPOTHESIS: Replacing per-image indexed crops with balanced batch-shared crops will finish verification while retaining at least 9,252 correct predictions because it preserves the 5:2:2:2:2 translation exposure distribution and removes costly advanced indexing from every training step.

INTENDED_EDIT: Use the training-step index to cycle through the same thirteen translation outcomes, applying each batch’s translation with a contiguous slice while preserving independent horizontal flips and the verified model and ensemble.

EVIDENCE: The verified design achieved 9,252 correct but took 76.9 training seconds; even single-view evaluation timed out later, pointing to training-path cost, while prior compute reductions left the per-example advanced-index augmentation unchanged.

<<<<<<< SEARCH
    del step, total_steps
    batch = images.shape[0]

    flip_mask = torch.rand(batch, 1, 1, 1, device=images.device) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)

    padding = 1
    padded = F.pad(images, (padding, padding, padding, padding), mode="replicate")
    height, width = images.shape[-2:]
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
    return images, labels
=======
    del total_steps
    batch = images.shape[0]

    flip_mask = torch.rand(batch, 1, 1, 1, device=images.device) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)

    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    height, width = images.shape[-2:]
    offset_draw = step % 13
    if offset_draw < 5:
        offset_y, offset_x = 1, 1
    elif offset_draw < 7:
        offset_y, offset_x = 0, 1
    elif offset_draw < 9:
        offset_y, offset_x = 2, 1
    elif offset_draw < 11:
        offset_y, offset_x = 1, 0
    else:
        offset_y, offset_x = 1, 2
    images = padded[
        ..., offset_y : offset_y + height, offset_x : offset_x + width
    ].contiguous()
    return images, labels
>>>>>>> REPLACE