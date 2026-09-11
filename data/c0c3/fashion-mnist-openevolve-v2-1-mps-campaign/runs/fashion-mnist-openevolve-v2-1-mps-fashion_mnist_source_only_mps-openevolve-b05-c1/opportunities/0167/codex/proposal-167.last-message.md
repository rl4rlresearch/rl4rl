MECHANISM: Low-rate diagonal translation augmentation

HYPOTHESIS: Reallocating one eighth of off-center training crops to diagonal one-pixel translations will exceed 9,360 correct predictions by improving two-axis pooling-phase robustness without changing the verified model, evaluation ensemble, or augmentation strength.

INTENDED_EDIT: Preserve the one-third centered-crop probability and two-thirds shifted-crop probability, while replacing one eighth of cardinal shifts with uniformly distributed diagonal shifts.

EVIDENCE: The diagonal evaluation ensemble and learned anti-aliased pooling both targeted phase sensitivity but timed out, while broader architecture, loss, and EMA changes reduced accuracy; training on sparse diagonal views tests the same mechanism without additional inference cost.

<<<<<<< SEARCH
    view_ids = torch.randint(0, 6, (batch_size,), device=images.device)
    offset_table = torch.tensor(
        ((1, 1), (1, 1), (0, 1), (2, 1), (1, 0), (1, 2)),
        device=images.device,
    )
=======
    view_ids = torch.randint(0, 48, (batch_size,), device=images.device)
    offset_table = torch.tensor(
        ((1, 1),) * 16
        + ((0, 1),) * 7
        + ((2, 1),) * 7
        + ((1, 0),) * 7
        + ((1, 2),) * 7
        + ((0, 0), (0, 2), (2, 0), (2, 2)),
        device=images.device,
    )
>>>>>>> REPLACE