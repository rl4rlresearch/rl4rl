MECHANISM: Center-anchored logit-space test-time aggregation

HYPOTHESIS: Increasing the unshifted view’s weight from 3 to 4 will exceed 9,166 correct predictions by retaining useful translation averaging while reducing dilution from less validation-aligned shifted views.

INTENDED_EDIT: Preserve the existing ten-view logit ensemble but modestly increase the relative weight of the centered image and its horizontal flip.

EVIDENCE: Adding diagonal shifted views reduced correctness from 9,166 to 9,155, while probability-space aggregation reached only 9,163; this supports keeping the established views and logit averaging while anchoring them more strongly to the centered input.

<<<<<<< SEARCH
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (4.0, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE