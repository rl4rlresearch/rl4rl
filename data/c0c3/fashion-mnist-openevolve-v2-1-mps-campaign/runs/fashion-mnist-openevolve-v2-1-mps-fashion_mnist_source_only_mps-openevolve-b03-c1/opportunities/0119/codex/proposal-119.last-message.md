MECHANISM: Low-overhead balanced cardinalization

HYPOTHESIS: Eliminating diagonal training translations while preserving balanced one-pixel cardinal shifts will exceed 9,196 correct predictions by aligning augmentation support with the successful validation views.

INTENDED_EDIT: For diagonally shifted samples, deterministically alternate which axis is recentered using batch-index parity, producing 25% centered and 75% balanced cardinal crops without additional random sampling.

EVIDENCE: Restricting translations to ±1 improved correctness from 9,166 to 9,196, while diagonal validation views reduced correctness to 9,155. Earlier diagonal-removal implementations timed out, motivating this computationally lean formulation.

<<<<<<< SEARCH
    offset_x = (
        torch.randint(0, 2, (images.shape[0],), device=images.device)
        + torch.randint(0, 2, (images.shape[0],), device=images.device)
        + 1
    )
    patches = padded.unfold(2, 28, 1).unfold(3, 28, 1)
=======
    offset_x = (
        torch.randint(0, 2, (images.shape[0],), device=images.device)
        + torch.randint(0, 2, (images.shape[0],), device=images.device)
        + 1
    )
    diagonal_mask = (offset_y != 2) & (offset_x != 2)
    center_x_mask = diagonal_mask & ((batch_indices & 1) == 0)
    offset_x.masked_fill_(center_x_mask, 2)
    offset_y.masked_fill_(diagonal_mask & ~center_x_mask, 2)
    patches = padded.unfold(2, 28, 1).unfold(3, 28, 1)
>>>>>>> REPLACE