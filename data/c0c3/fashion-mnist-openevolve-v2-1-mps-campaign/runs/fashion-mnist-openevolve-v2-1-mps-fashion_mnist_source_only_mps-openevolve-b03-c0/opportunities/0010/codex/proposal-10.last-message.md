MECHANISM: Translation-only spatial augmentation

HYPOTHESIS: Independent two-pixel translations without reflection will exceed 9,290 correct predictions by improving position robustness while avoiding the strongly harmful orientation change.

INTENDED_EDIT: Replace unaugmented batches with per-image random crops from two-pixel replicated padding; retain the verified architecture, batch size, optimizer, loss, and schedule.

EVIDENCE: Horizontal-reflection test-time averaging reduced correctness from 9,290 to 8,663, indicating reflection is not label-preserving for this task; therefore, the earlier combined translation-and-reflection regression does not rule out translation alone.

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    padding = 2
    padded = F.pad(images, (padding, padding, padding, padding), mode="replicate")
    batch_size, channels, height, width = images.shape

    row_offsets = torch.randint(
        0, 2 * padding + 1, (batch_size,), device=images.device
    )
    column_offsets = torch.randint(
        0, 2 * padding + 1, (batch_size,), device=images.device
    )
    rows = (
        torch.arange(height, device=images.device).unsqueeze(0)
        + row_offsets.unsqueeze(1)
    )
    columns = (
        torch.arange(width, device=images.device).unsqueeze(0)
        + column_offsets.unsqueeze(1)
    )

    shifted = padded.gather(
        2,
        rows[:, None, :, None].expand(
            batch_size, channels, height, padded.shape[-1]
        ),
    )
    shifted = shifted.gather(
        3,
        columns[:, None, None, :].expand(
            batch_size, channels, height, width
        ),
    )
    return shifted, labels
>>>>>>> REPLACE