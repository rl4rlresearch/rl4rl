MECHANISM: Stratified translation-orbit sampling

HYPOTHESIS: Replacing multinomial translation draws with randomly permuted systematic samples will exceed 9,237 correct predictions by preserving the proven translation distribution while reducing per-batch augmentation and BatchNorm noise.

INTENDED_EDIT: Sample nearly exact reliability-weighted translation proportions in every minibatch, while randomly assigning offsets to examples.

EVIDENCE: Reliability-weighted translation sampling contributed to the 9,186-correct flip-complete design and remains in the 9,237-correct best design; with only 522 updates, reducing multinomial count variance is a targeted refinement of that proven invariance recipe.

<<<<<<< SEARCH
    flat_offsets = torch.multinomial(
        translation_weights, images.size(0), replacement=True
    )
=======
    sample_positions = (
        torch.arange(
            images.size(0),
            device=images.device,
            dtype=translation_weights.dtype,
        )
        + torch.rand(
            (),
            device=images.device,
            dtype=translation_weights.dtype,
        )
    ) / images.size(0)
    flat_offsets = torch.searchsorted(
        translation_weights.cumsum(dim=0),
        sample_positions,
    )
    flat_offsets = flat_offsets[
        torch.randperm(images.size(0), device=images.device)
    ]
>>>>>>> REPLACE