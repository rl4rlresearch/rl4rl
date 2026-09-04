MECHANISM: Progressive canonical-view micro-upweighting

HYPOTHESIS: Raising each unshifted view’s weight to 1.125 will retain 9,290 correct predictions while reducing validation cross-entropy below 0.20246240730285645.

INTENDED_EDIT: Increase the original and horizontally flipped original views’ TTA weight from 1.0625× to 1.125×, leaving shifted views and all other calibration unchanged.

EVIDENCE: Moving from 1.0× to 1.0625× preserved 9,290 correct and improved cross-entropy from 0.20248969497680663 to 0.20246240730285645, while 1.5× improved cross-entropy further but lost accuracy; 1.125× is the next informative step toward that tradeoff boundary.

<<<<<<< SEARCH
        view_weights = (1.0625, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.125, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE