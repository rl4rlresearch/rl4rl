MECHANISM: Full 3×3 translation-and-flip test-time ensemble

HYPOTHESIS: Adding the four missing diagonal one-pixel views will exceed 9,159 correct predictions by more completely averaging the local translations that improved the prior two-view ensemble by 25 images.

INTENDED_EDIT: Expand validation inference from five spatial views and their flips to all nine offsets in the centered 3×3 translation neighborhood and their flips.

EVIDENCE: The prior augmentation-matched cardinal-translation ensemble improved validation correctness from 9,134 to 9,159 without changing training; diagonal offsets complete the same one-pixel invariance neighborhood.

<<<<<<< SEARCH
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
=======
        views = (
            images,
            padded[:, :, 0:28, 0:28],
            padded[:, :, 0:28, 1:29],
            padded[:, :, 0:28, 2:30],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
            padded[:, :, 2:30, 0:28],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 2:30, 2:30],
        )
>>>>>>> REPLACE