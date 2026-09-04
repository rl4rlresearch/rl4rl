MECHANISM: Short-warmup ensemble optimization

HYPOTHESIS: Restoring translation-free inputs and shortening warmup from 5% to 2% will exceed 9,330 correct predictions by providing more peak-rate optimization within the fixed exposure while preserving the winning late-training schedule.

INTENDED_EDIT: Remove the harmful random translations from Reference Design 1 and shorten only its learning-rate warmup.

EVIDENCE: Translation augmentation reduced the winning design from 9,330 to 9,222 correct, while changing the late cosine floor also regressed; this motivates restoring the winner and testing the untouched early learning-rate allocation.

<<<<<<< SEARCH
    del step, total_steps
    padding = 2
    height, width = images.shape[-2:]
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )

    batch_size, channels = images.shape[:2]
    top = torch.randint(
        0,
        2 * padding + 1,
        (batch_size,),
        device=images.device,
    )
    left = torch.randint(
        0,
        2 * padding + 1,
        (batch_size,),
        device=images.device,
    )

    rows = torch.arange(height, device=images.device).unsqueeze(0)
    rows = rows + top.unsqueeze(1)
    translated = padded.gather(
        2,
        rows[:, None, :, None].expand(
            batch_size,
            channels,
            height,
            padded.shape[-1],
        ),
    )

    columns = torch.arange(width, device=images.device).unsqueeze(0)
    columns = columns + left.unsqueeze(1)
    translated = translated.gather(
        3,
        columns[:, None, None, :].expand(
            batch_size,
            channels,
            height,
            width,
        ),
    )
    return translated, labels
=======
    del step, total_steps
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    warmup_steps = max(1, int(0.05 * total_steps))
=======
    warmup_steps = max(1, int(0.02 * total_steps))
>>>>>>> REPLACE