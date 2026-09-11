MECHANISM: Stratified per-image translation sampling

HYPOTHESIS: Enforcing the 5:2:2:2:2 translation ratio within each batch while retaining randomized per-image offsets will exceed 9,252 correct predictions by reducing augmentation-induced gradient variance without sacrificing crop diversity.

INTENDED_EDIT: Replace independent offset draws with shuffled 13-example strata containing five centered crops and two of each directional crop; any partial stratum is an unbiased random subset.

EVIDENCE: Batch-shared translations completed but lost 34 correct predictions, showing per-image crop diversity matters; this preserves that diversity and the verified translation distribution while improving batch-level balance.

<<<<<<< SEARCH
    offset_draw = torch.randint(0, 13, (batch,), device=images.device)
=======
    offset_pattern = torch.randperm(13, device=images.device)
    full_strata, remainder = divmod(batch, 13)
    offset_draw = offset_pattern.repeat(full_strata)
    if remainder:
        offset_draw = torch.cat((offset_draw, offset_pattern[:remainder]))
    offset_draw = offset_draw[torch.randperm(batch, device=images.device)]
>>>>>>> REPLACE