MECHANISM: Training-matched test-time view weighting

HYPOTHESIS: Assigning exactly 25% of ensemble weight to centered views will exceed 9,210 correct predictions by matching the augmentation balance whose higher and lower center rates both performed worse.

INTENDED_EDIT: Change center-to-cardinal validation weights from 3:2:2:2:2 to 4:3:3:3:3 while retaining arithmetic probability pooling.

EVIDENCE: The 25%-center training distribution reached 9,206 correct, versus 9,198 at 27.3% and 9,196 at 22.7%; the prior verification of this exact low-cost weighting timed out without yielding subject-level evidence.

<<<<<<< SEARCH
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (4.0, 3.0, 3.0, 3.0, 3.0)
>>>>>>> REPLACE