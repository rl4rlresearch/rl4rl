MECHANISM: Local interpolation of center-dominant axial ensembling

HYPOTHESIS: A center weight of 2.75 will exceed 9,168 correct predictions by refining the unexplored interval below weight 3, while weights above 3 reduced accuracy.

INTENDED_EDIT: Reduce the centered crop’s ensemble weight from 4.0 to 2.75 while retaining the four axial crops and mirrored views.

EVIDENCE: Weight 3 achieved the best result at 9,168 correct; weight 2 reached 9,163, while weights 3.25 and 4 both declined to 9,166, locating the observed optimum at or just below 3.

<<<<<<< SEARCH
        crop_weights = (4.0, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (2.75, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE