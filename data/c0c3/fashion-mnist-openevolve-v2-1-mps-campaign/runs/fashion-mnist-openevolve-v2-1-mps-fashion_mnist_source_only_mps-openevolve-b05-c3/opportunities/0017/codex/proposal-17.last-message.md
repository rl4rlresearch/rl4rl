MECHANISM: Incrementally extended inference-aligned augmentation

HYPOTHESIS: Beginning center/cardinal augmentation after 19/32 of training will exceed 9,167 correct predictions by continuing the observed monotonic benefit of longer inference-aligned phases while changing the validated boundary by only 1/64 of training.

INTENDED_EDIT: Use broad ±2 translations for the first 19/32 of training and center/cardinal one-pixel translations for the final 13/32, retaining all other validated settings.

EVIDENCE: Extending the cardinal phase from 1/4 to 1/3 to 3/8 to 25/64 increased validation correct from 9,141 to 9,155 to 9,163 to 9,167 and most recently reduced cross-entropy to 0.23885; 13/32 is the next conservative boundary extension.

<<<<<<< SEARCH
    if step * 64 < total_steps * 39:
=======
    if step * 32 < total_steps * 19:
>>>>>>> REPLACE