MECHANISM: Slightly center-biased probability ensemble

HYPOTHESIS: Increasing centered-view weight from 27.3% to 28.9% will exceed 9,210 correct predictions by continuing the improvement observed when center weight increased from 25% to 27.3%.

INTENDED_EDIT: Change center-to-cardinal validation weights from 3:2:2:2:2 to 3.25:2:2:2:2 while preserving arithmetic probability pooling.

EVIDENCE: Arithmetic probability pooling with 27.3% centered weight achieved 9,210 correct, while reducing centered weight to 25% achieved 9,205; a small upward adjustment directly tests the supported direction.

<<<<<<< SEARCH
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.25, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE