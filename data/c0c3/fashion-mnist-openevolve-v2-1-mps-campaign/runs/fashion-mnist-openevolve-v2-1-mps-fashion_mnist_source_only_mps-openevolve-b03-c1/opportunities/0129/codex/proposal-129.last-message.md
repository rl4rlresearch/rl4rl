MECHANISM: Training-matched test-time view weighting

HYPOTHESIS: Weighting centered validation views at exactly 25% will exceed 9,210 correct predictions by matching the augmentation distribution whose higher and lower centered-crop variants both performed worse.

INTENDED_EDIT: Change center-to-cardinal validation weights from 3:2:2:2:2 to 4:3:3:3:3 while retaining arithmetic probability pooling and all ten views.

EVIDENCE: The 25%-center training distribution achieved 9,206 correct before probability pooling, while increasing center exposure to 27.3% produced 9,198 and decreasing it to 22.7% produced 9,196; the current validation weights assign 27.3% to centered views.

<<<<<<< SEARCH
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (4.0, 3.0, 3.0, 3.0, 3.0)
>>>>>>> REPLACE