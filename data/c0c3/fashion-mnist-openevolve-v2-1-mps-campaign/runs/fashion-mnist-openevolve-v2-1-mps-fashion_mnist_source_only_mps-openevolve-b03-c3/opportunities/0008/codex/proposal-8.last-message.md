MECHANISM: Reliability-weighted translation augmentation

HYPOTHESIS: Sampling training translations with the proven 5×5 evaluation weights will exceed 9,051 correct predictions by retaining radius-two invariance while focusing exposure on the more reliable center and radius-one views.

INTENDED_EDIT: Replace uniform 5×5 translation sampling with the exact center/cardinal/diagonal/radius-two distribution used by the best verified logit ensemble.

EVIDENCE: The weighted 5×5 logit ensemble improved from 9,042 to 9,051 correct, showing radius-two views help when lightly weighted; uniform training currently assigns them 64% of examples despite their 10% evaluation weight.

<<<<<<< SEARCH
    offsets = torch.randint(0, 5, (images.size(0), 2), device=images.device)
=======
    translation_weights = images.new_tensor(
        (
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.09000, 0.36000, 0.09000, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
        )
    )
    flat_offsets = torch.multinomial(
        translation_weights, images.size(0), replacement=True
    )
    offsets = torch.stack(
        (flat_offsets.div(5, rounding_mode="floor"), flat_offsets.remainder(5)),
        dim=1,
    )
>>>>>>> REPLACE