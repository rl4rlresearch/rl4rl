MECHANISM: Increased finite-exposure optimization granularity

HYPOTHESIS: Reducing the batch size to 128 will exceed 9,284 correct predictions by increasing optimizer updates from 522 to approximately 782 within the fixed 100,000-example budget, while the paired-image loss still gives BatchNorm an effective 256-image batch.

INTENDED_EDIT: Change only the training batch size from 192 to 128, preserving the validated architecture, loss, augmentation, TTA, and calibration.

EVIDENCE: Calibration is saturated at 9,284 correct, while channel attention, aligned translation sampling, probability-space TTA, and late EMA all reduced accuracy; batch granularity remains an untested way to improve convergence without altering the established classifier computation.

<<<<<<< SEARCH
BATCH_SIZE = 192
=======
BATCH_SIZE = 128
>>>>>>> REPLACE