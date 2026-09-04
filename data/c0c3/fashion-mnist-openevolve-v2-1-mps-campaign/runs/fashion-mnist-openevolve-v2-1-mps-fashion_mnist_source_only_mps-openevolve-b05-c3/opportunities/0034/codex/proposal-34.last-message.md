MECHANISM: Validated inference-aligned augmentation curriculum

HYPOTHESIS: Moving the broad-to-cardinal transition from 5/8 to 39/64 will increase validation correct from 9,163 to at least the previously observed 9,167.

INTENDED_EDIT: Restore the best verified 39/64 augmentation boundary while preserving the architecture, optimizer, EMA, loss, and inference ensemble.

EVIDENCE: Reference Design 1 achieved 9,167 correct and 0.23885 cross-entropy; otherwise identical later transitions scored 9,163 at 5/8, 9,155 at 2/3, and 9,141 at 3/4.

<<<<<<< SEARCH
    if step * 8 < total_steps * 5:
=======
    if step * 64 < total_steps * 39:
>>>>>>> REPLACE