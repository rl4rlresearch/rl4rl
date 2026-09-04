MECHANISM: Bracketed reverse-direction vertical TTA refinement

HYPOTHESIS: Moving four float32 ULPs from the second vertical crop to the first relative to Reference Design 1 will retain 9,348 correct predictions while reducing validation cross-entropy below 0.1877065631866455.

INTENDED_EDIT: Set the vertical crop weights to the previously timed-out reverse half-step around the best verified weighting, preserving their combined and total ensemble weight.

EVIDENCE: Reference Design 1 has the lowest verified cross-entropy; both tested forward offsets were worse, while the symmetric reverse offset timed out and produced no contrary validation evidence.

<<<<<<< SEARCH
            1.31201267242431640625,
            1.31494045257568359375,
=======
            1.312014102935791015625,
            1.314939022064208984375,
>>>>>>> REPLACE