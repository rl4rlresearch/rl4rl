MECHANISM: Moderately stronger center-biased cross translation augmentation

HYPOTHESIS: Increasing centered-crop probability from 5/13 to 3/7 while retaining only cardinal translations will exceed 9,249 correct predictions by continuing the demonstrated benefit of modestly reducing translation magnitude without reaching the previously harmful 4/9 center concentration.

INTENDED_EDIT: Change cross-shaped crop weights from 5:2 center-to-cardinal to 3:1, giving probability 3/7 to the centered crop and 1/7 to each cardinal shift.

EVIDENCE: Increasing the cardinal-only center weight from 2:1 to 5:2 improved validation_correct from 9,232 to 9,238, while removing diagonal inference views subsequently raised it to 9,249; 3:1 is the next controlled center-weight increase and remains below the unsuccessful 4/9 centered probability.

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
    offset_draw = torch.randint(0, 7, (batch,), device=images.device)
    offsets_y = (
        1
        + ((offset_draw >= 4) & (offset_draw < 5)).long()
        - ((offset_draw >= 3) & (offset_draw < 4)).long()
    )
    offsets_x = (
        1
        + ((offset_draw >= 6) & (offset_draw < 7)).long()
        - ((offset_draw >= 5) & (offset_draw < 6)).long()
    )
>>>>>>> REPLACE