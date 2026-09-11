MECHANISM: Inference-distribution-matched geometric augmentation

HYPOTHESIS: Sampling only centered and cardinal one-pixel translations, with the center sampled twice, will exceed 9,175 correct predictions by removing harmful diagonal exposure and matching the successful center-weighted validation ensemble.

INTENDED_EDIT: Replace uniform sampling over all nine ±1 translations with a six-entry distribution containing two centered and four cardinal offsets.

EVIDENCE: Restricting training from ±2 to ±1 improved correctness from 9,162 to 9,175; diagonal validation views regressed, while doubling the centered-view weight improved inference, motivating the same center/cardinal distribution during training.

<<<<<<< SEARCH
    offsets = torch.randint(0, 3, (batch_size, 2), device=images.device)
    batch_indices = torch.arange(batch_size, device=images.device)
=======
    view_ids = torch.randint(0, 6, (batch_size,), device=images.device)
    offset_table = torch.tensor(
        ((1, 1), (1, 1), (0, 1), (2, 1), (1, 0), (1, 2)),
        device=images.device,
    )
    offsets = offset_table[view_ids]
    batch_indices = torch.arange(batch_size, device=images.device)
>>>>>>> REPLACE