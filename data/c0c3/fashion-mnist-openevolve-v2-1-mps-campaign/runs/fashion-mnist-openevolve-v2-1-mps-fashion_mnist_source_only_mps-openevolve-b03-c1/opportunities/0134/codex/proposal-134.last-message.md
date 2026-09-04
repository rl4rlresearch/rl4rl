MECHANISM: Fine-grained centered-view reweighting

HYPOTHESIS: Increasing centered-view weight from 27.3% to 28.1% will exceed 9,210 correct predictions by continuing the improvement observed from 25% to 27.3% without making the larger unverified jump to 28.9%.

INTENDED_EDIT: Change the centered validation-view weight from 3.0 to 3.125 while preserving arithmetic probability pooling and all training behavior.

EVIDENCE: Arithmetic probability pooling achieved 9,210 correct at 27.3% centered weight, whereas 25% achieved 9,205; the 28.9% test timed out, motivating a smaller low-cost step in the supported direction.

<<<<<<< SEARCH
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.125, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE