MECHANISM: Validation-ratio cardinalization

HYPOTHESIS: Centering one-eleventh of diagonal candidates before cardinalizing the remainder will exceed 9,206 correct predictions by matching the successful validation ensemble’s 3:8 center-to-cardinal exposure ratio while preserving balanced one-pixel shifts.

INTENDED_EDIT: Convert one-eleventh of sampled diagonal shifts into centered crops using a step-rotated batch-index mask; deterministically cardinalize all remaining diagonals as before.

EVIDENCE: Eliminating diagonal translations improved correctness from 9,196 to 9,206. The current fast implementation yields a 1:3 center-to-cardinal ratio, while the successful validation ensemble uses 3:8; centering one-eleventh of diagonal candidates closes that specific distribution gap without the costly deterministic-cycle implementation that timed out.

<<<<<<< SEARCH
    del step, total_steps
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
=======
    del total_steps
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
>>>>>>> REPLACE

<<<<<<< SEARCH
    diagonal_mask = (offset_y != 2) & (offset_x != 2)
    center_x_mask = diagonal_mask & ((batch_indices & 1) == 0)
    offset_x.masked_fill_(center_x_mask, 2)
    offset_y.masked_fill_(diagonal_mask & ~center_x_mask, 2)
=======
    diagonal_mask = (offset_y != 2) & (offset_x != 2)
    center_diagonal_mask = diagonal_mask & (
        ((batch_indices + step) % 11) == 0
    )
    offset_x.masked_fill_(center_diagonal_mask, 2)
    offset_y.masked_fill_(center_diagonal_mask, 2)
    cardinal_mask = diagonal_mask & ~center_diagonal_mask
    center_x_mask = cardinal_mask & ((batch_indices & 1) == 0)
    offset_x.masked_fill_(center_x_mask, 2)
    offset_y.masked_fill_(cardinal_mask & ~center_x_mask, 2)
>>>>>>> REPLACE