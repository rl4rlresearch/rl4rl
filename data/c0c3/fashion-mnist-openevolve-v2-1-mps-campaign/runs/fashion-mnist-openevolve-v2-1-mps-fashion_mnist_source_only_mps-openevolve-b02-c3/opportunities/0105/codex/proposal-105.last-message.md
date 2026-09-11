MECHANISM: Directional vertical-shift TTA reweighting

HYPOTHESIS: Redistributing 1/1024 ensemble weight from the downward-shifted crop to the upward-shifted crop will exceed 9,348 correct predictions or retain 9,348 while lowering cross-entropy.

INTENDED_EDIT: Break the unexplored symmetry between vertical crops while preserving their combined weight, center weight, horizontal weights, and temperature.

EVIDENCE: Favoring vertical over horizontal translations improved correctness from 9,344 to 9,348, but symmetric vertical-weight refinement reached a correctness boundary; directional vertical weighting is the nearest untested TTA axis.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.3134765625,
            1.3134765625,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.3125,
            1.314453125,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE