MECHANISM: Further higher-update residual refinement

HYPOTHESIS: Reducing the unique-image batch size from 64 to 32 will exceed 9,312 correct validation predictions by doubling optimizer updates while retaining an effective 64-view paired-flip training batch.

INTENDED_EDIT: Keep the best verified architecture, paired-flip objective, optimizer, and schedule unchanged, and reduce only `BATCH_SIZE` from 64 to 32.

EVIDENCE: Reducing batch size from 128 to 64 improved the same 249,393-parameter implementation from 9,310 to 9,312 correct and lowered cross-entropy from 0.20765 to 0.20389, directly motivating another isolated increase in update frequency.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0
=======
BATCH_SIZE = 32
GRAD_CLIP_NORM = 5.0
>>>>>>> REPLACE