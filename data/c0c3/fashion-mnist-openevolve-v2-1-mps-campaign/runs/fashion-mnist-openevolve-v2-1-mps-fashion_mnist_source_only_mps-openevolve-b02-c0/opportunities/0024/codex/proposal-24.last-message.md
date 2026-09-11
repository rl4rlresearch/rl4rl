MECHANISM: Cardinal-only translation-and-flip ensemble

HYPOTHESIS: Removing diagonal inference views will exceed 9,238 correct predictions by aligning evaluation with the successful cardinal-only crop distribution and avoiding weak out-of-distribution diagonal shifts.

INTENDED_EDIT: Restrict validation averaging from the full 3×3 translation grid to the centered image and four one-pixel cardinal translations, retaining horizontal-flip averaging.

EVIDENCE: Removing diagonal crops from training improved validation_correct from 9,221 to 9,232, while adding diagonal inference views previously contributed only one correct prediction; the current cardinal-only training distribution therefore makes cardinal-only inference the most targeted next test.

<<<<<<< SEARCH
        views = (
            images,
            padded[..., :height, :width],
            padded[..., :height, 1 : width + 1],
            padded[..., :height, 2 : width + 2],
            padded[..., 1 : height + 1, :width],
            padded[..., 1 : height + 1, 2 : width + 2],
            padded[..., 2 : height + 2, :width],
            padded[..., 2 : height + 2, 1 : width + 1],
            padded[..., 2 : height + 2, 2 : width + 2],
        )
=======
        views = (
            images,
            padded[..., :height, 1 : width + 1],
            padded[..., 1 : height + 1, :width],
            padded[..., 1 : height + 1, 2 : width + 2],
            padded[..., 2 : height + 2, 1 : width + 1],
        )
>>>>>>> REPLACE