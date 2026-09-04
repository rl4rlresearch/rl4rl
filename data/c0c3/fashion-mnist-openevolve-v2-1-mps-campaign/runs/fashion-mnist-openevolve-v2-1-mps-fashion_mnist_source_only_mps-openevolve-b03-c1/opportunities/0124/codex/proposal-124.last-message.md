MECHANISM: Step-rotated balanced cardinalization

HYPOTHESIS: Alternating the recentered axis across optimizer steps will exceed 9,206 correct predictions by removing fixed batch-position coupling while preserving the successful 25% center/75% balanced-cardinal augmentation distribution.

INTENDED_EDIT: Include step parity in the existing low-overhead diagonal cardinalization rule; leave augmentation frequencies, randomness, architecture, and optimization unchanged.

EVIDENCE: Eliminating diagonal shifts with batch-parity cardinalization improved correctness from 9,196 to 9,206, while moving center exposure in either direction reduced it to 9,198 and 9,196; this motivates retaining the winning distribution and isolating its fixed-position artifact.

<<<<<<< SEARCH
    del step, total_steps
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
=======
    del total_steps
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
>>>>>>> REPLACE

<<<<<<< SEARCH
    center_x_mask = diagonal_mask & ((batch_indices & 1) == 0)
=======
    center_x_mask = diagonal_mask & (((batch_indices + step) & 1) == 0)
>>>>>>> REPLACE