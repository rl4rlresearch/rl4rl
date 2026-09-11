MECHANISM: Reliability-matched axial augmentation

HYPOTHESIS: Training on the same center-weighted axial translation distribution used by the best inference ensemble will exceed 9,168 correct predictions by avoiding the less reliable diagonal and two-pixel translations.

INTENDED_EDIT: Use the qualified center weight of 3.0 and replace uniform independent ±2-pixel training translations with center-three-times-plus-four-axial ±1-pixel sampling.

EVIDENCE: Center weight 3 achieved the best observed 9,168 correct, while adding diagonal inference views reduced accuracy to 9,159; the current training sampler nevertheless emphasizes diagonal and two-pixel transformations absent from the successful ensemble.

<<<<<<< SEARCH
        crop_weights = (2.0, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    padding = 2
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    offsets_y = torch.randint(
        0, 2 * padding + 1, (batch, 1), device=images.device
    )
    offsets_x = torch.randint(
        0, 2 * padding + 1, (batch, 1), device=images.device
    )
=======
    padding = 1
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    translation = torch.randint(0, 7, (batch,), device=images.device)
    offsets_y = torch.tensor(
        (1, 1, 1, 0, 2, 1, 1), device=images.device
    )[translation].unsqueeze(1)
    offsets_x = torch.tensor(
        (1, 1, 1, 1, 1, 0, 2), device=images.device
    )[translation].unsqueeze(1)
>>>>>>> REPLACE