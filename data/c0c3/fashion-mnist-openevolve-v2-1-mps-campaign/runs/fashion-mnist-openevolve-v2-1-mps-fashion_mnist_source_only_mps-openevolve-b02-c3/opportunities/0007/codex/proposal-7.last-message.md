MECHANISM: Local interpolation of center-dominant test-time ensembling

HYPOTHESIS: A center weight of 3.25 will exceed 9,168 correct predictions by refining the apparent optimum between weight 3, which achieved 9,168, and weight 4, which declined to 9,166.

INTENDED_EDIT: Increase the centered crop’s ensemble weight from 3.0 to 3.25 while retaining the four axial crops and mirrored views.

EVIDENCE: Center weights 1, 2, 3, and 4 produced 9,162, 9,163, 9,168, and 9,166 correct predictions respectively, locating the best observed region near weight 3 with diminishing performance before weight 4.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (3.25, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE