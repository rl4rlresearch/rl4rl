MECHANISM: Center-weighted one-pixel translation augmentation

HYPOTHESIS: Biasing one-pixel crops toward the centered position will exceed 9,203 correct predictions by further reducing training–validation translation mismatch while retaining useful local translation robustness.

INTENDED_EDIT: Replace uniform per-axis crop offsets with a symmetric 1:2:1 distribution, increasing the centered crop probability from 1/9 to 1/4 without changing the translation range.

EVIDENCE: Reducing augmentation from uniform two-pixel shifts to uniform one-pixel shifts improved validation_correct from 9,157 to 9,203, strongly motivating a further controlled reduction in average translation magnitude.

<<<<<<< SEARCH
    offsets_y = torch.randint(0, 2 * padding + 1, (batch,), device=images.device)
    offsets_x = torch.randint(0, 2 * padding + 1, (batch,), device=images.device)
=======
    offset_draw_y = torch.randint(0, 4, (batch,), device=images.device)
    offset_draw_x = torch.randint(0, 4, (batch,), device=images.device)
    offsets_y = (offset_draw_y >= 1).long() + (offset_draw_y == 3).long()
    offsets_x = (offset_draw_x >= 1).long() + (offset_draw_x == 3).long()
>>>>>>> REPLACE