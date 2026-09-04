MECHANISM: Moderately center-biased cross translation augmentation

HYPOTHESIS: Raising centered-crop probability from 1/3 to 5/13 while retaining only cardinal translations will exceed 9,232 correct predictions by modestly reducing translation mismatch without the excessive center concentration of 1:4:1 sampling.

INTENDED_EDIT: Change cross-shaped crop weights from 2:1 center-to-cardinal to 5:2, giving probability 5/13 to center and 2/13 to each cardinal shift.

EVIDENCE: Removing diagonal crops improved validation_correct from 9,221 to 9,232, while earlier moderate center weighting improved 9,203 to 9,221 and stronger 1:4:1 weighting fell slightly; this motivates a smaller center-weight increase within the successful cross-shaped distribution.

<<<<<<< SEARCH
    offset_draw = torch.randint(0, 6, (batch,), device=images.device)
    offsets_y = (
        1
        + (offset_draw == 3).long()
        - (offset_draw == 2).long()
    )
    offsets_x = (
        1
        + (offset_draw == 5).long()
        - (offset_draw == 4).long()
    )
=======
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
>>>>>>> REPLACE