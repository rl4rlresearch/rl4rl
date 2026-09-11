MECHANISM: Orientation-aware translation ensembling

HYPOTHESIS: Weighting horizontal shifts more than vertical shifts will exceed 9,344 correct predictions because paired horizontal-flip supervision makes horizontal views more reliable, while the large regressions from removing spatial organization show that vertical alignment remains class-relevant.

INTENDED_EDIT: Preserve the proven center weight, EMA, and total TTA weight while reallocating 0.25 weight from each vertical crop to each horizontal crop.

EVIDENCE: The weight-3 symmetric ensemble achieved the best 9,344-correct result, whereas changing its overall center/shift balance regressed; this motivates holding that balance fixed and testing orientation-specific reliability instead.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (3.0, 0.75, 0.75, 1.25, 1.25)
>>>>>>> REPLACE