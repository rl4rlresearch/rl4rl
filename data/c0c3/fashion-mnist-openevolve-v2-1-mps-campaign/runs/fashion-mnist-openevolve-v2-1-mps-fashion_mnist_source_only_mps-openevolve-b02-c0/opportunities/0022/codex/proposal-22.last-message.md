MECHANISM: Cross-shaped one-pixel crop augmentation

HYPOTHESIS: Reallocating diagonal crop probability to centered and cardinal crops will exceed 9,221 correct predictions by preserving useful translation robustness while reducing exposure to the least valuable shifts.

INTENDED_EDIT: Replace independent per-axis 1:2:1 crop sampling with a joint distribution assigning probability 1/3 to the centered crop, 1/6 to each cardinal shift, and zero to diagonal shifts.

EVIDENCE: Cardinal translation inference improved validation_correct from 9,146 to 9,156, while adding all four diagonal views gained only one additional correct prediction; meanwhile, increasing centered training exposure improved the best result to 9,221.

<<<<<<< SEARCH
    offset_draw_y = torch.randint(0, 4, (batch,), device=images.device)
    offset_draw_x = torch.randint(0, 4, (batch,), device=images.device)
    offsets_y = (offset_draw_y >= 1).long() + (offset_draw_y == 3).long()
    offsets_x = (offset_draw_x >= 1).long() + (offset_draw_x == 3).long()
=======
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
>>>>>>> REPLACE