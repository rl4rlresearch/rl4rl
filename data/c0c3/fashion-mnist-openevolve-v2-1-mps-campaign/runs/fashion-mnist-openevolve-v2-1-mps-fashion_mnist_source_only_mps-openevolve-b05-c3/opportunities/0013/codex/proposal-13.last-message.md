MECHANISM: Half-run inference-aligned augmentation

HYPOTHESIS: Beginning center/cardinal one-pixel augmentation halfway through training will exceed 9,155 correct predictions by further increasing inference-matched optimization while preserving 50,000 early examples of broad ±2 translation regularization.

INTENDED_EDIT: Extend the validated inference-aligned augmentation phase from the final third to the final half of training.

EVIDENCE: Extending this phase from the final quarter to the final third increased validation correct from 9,141 to 9,155 and reduced cross-entropy from 0.24300 to 0.24135, directly supporting another controlled boundary extension.

<<<<<<< SEARCH
    if step * 3 < total_steps * 2:
=======
    if step * 2 < total_steps:
>>>>>>> REPLACE