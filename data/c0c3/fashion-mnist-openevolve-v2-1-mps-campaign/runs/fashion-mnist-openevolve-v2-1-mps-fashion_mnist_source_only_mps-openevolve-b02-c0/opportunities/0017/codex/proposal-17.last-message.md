MECHANISM: Intermediate center-weighted translation augmentation

HYPOTHESIS: A per-axis 1:3:1 crop distribution will exceed 9,221 correct predictions by interpolating between the best 1:2:1 weighting and the slightly worse, more center-concentrated 1:4:1 weighting.

INTENDED_EDIT: Increase the centered per-axis crop probability from 1/2 to 3/5 while preserving one-pixel translations and every other training and inference setting.

EVIDENCE: The 1:2:1 distribution achieved 9,221 correct, while increasing directly to 1:4:1 produced 9,220; testing the unmeasured intermediate 1:3:1 distribution is the most targeted refinement of the demonstrated center-weighting effect.

<<<<<<< SEARCH
    offset_draw_y = torch.randint(0, 4, (batch,), device=images.device)
    offset_draw_x = torch.randint(0, 4, (batch,), device=images.device)
    offsets_y = (offset_draw_y >= 1).long() + (offset_draw_y == 3).long()
    offsets_x = (offset_draw_x >= 1).long() + (offset_draw_x == 3).long()
=======
    offset_draw_y = torch.randint(0, 5, (batch,), device=images.device)
    offset_draw_x = torch.randint(0, 5, (batch,), device=images.device)
    offsets_y = (offset_draw_y >= 1).long() + (offset_draw_y == 4).long()
    offsets_x = (offset_draw_x >= 1).long() + (offset_draw_x == 4).long()
>>>>>>> REPLACE