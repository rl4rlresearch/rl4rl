MECHANISM: Intermediate center-biased cross translation augmentation

HYPOTHESIS: Increasing centered-crop probability from 5/13 to 2/5 will exceed 9,249 correct predictions by continuing the observed benefit of modest center bias while remaining more conservative than the unverified 3/7 setting.

INTENDED_EDIT: Change cardinal-only crop weights from 5:2 to 8:3, assigning probability 2/5 to the centered crop and 3/20 to each cardinal translation.

EVIDENCE: Increasing center probability within cross-shaped augmentation from 1/3 to 5/13 improved validation_correct from 9,232 to 9,238; the next stronger 3/7 setting timed out, motivating this intermediate refinement.

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
=======
    offset_draw = torch.randint(0, 20, (batch,), device=images.device)
    offsets_y = (
        1
        + ((offset_draw >= 11) & (offset_draw < 14)).long()
        - ((offset_draw >= 8) & (offset_draw < 11)).long()
    )
    offsets_x = (
        1
        + ((offset_draw >= 17) & (offset_draw < 20)).long()
        - ((offset_draw >= 14) & (offset_draw < 17)).long()
    )
>>>>>>> REPLACE