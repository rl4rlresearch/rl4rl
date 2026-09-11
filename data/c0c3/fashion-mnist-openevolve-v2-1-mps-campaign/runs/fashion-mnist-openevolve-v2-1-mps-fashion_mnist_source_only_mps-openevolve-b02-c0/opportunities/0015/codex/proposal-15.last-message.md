MECHANISM: Stronger center-weighted translation augmentation

HYPOTHESIS: Increasing the per-axis crop distribution from 1:2:1 to 1:4:1 will exceed 9,221 correct predictions by further reducing training–validation translation mismatch while retaining occasional one-pixel robustness.

INTENDED_EDIT: Increase the centered crop probability from 1/4 to 4/9 while preserving the one-pixel translation range and all other training and inference settings.

EVIDENCE: Changing uniform one-pixel crops to a center-weighted 1:2:1 distribution improved validation_correct from 9,203 to 9,221, motivating a controlled further increase in center weighting.

<<<<<<< SEARCH
    offset_draw_y = torch.randint(0, 4, (batch,), device=images.device)
    offset_draw_x = torch.randint(0, 4, (batch,), device=images.device)
    offsets_y = (offset_draw_y >= 1).long() + (offset_draw_y == 3).long()
    offsets_x = (offset_draw_x >= 1).long() + (offset_draw_x == 3).long()
=======
    offset_draw_y = torch.randint(0, 6, (batch,), device=images.device)
    offset_draw_x = torch.randint(0, 6, (batch,), device=images.device)
    offsets_y = (offset_draw_y >= 1).long() + (offset_draw_y == 5).long()
    offsets_x = (offset_draw_x >= 1).long() + (offset_draw_x == 5).long()
>>>>>>> REPLACE