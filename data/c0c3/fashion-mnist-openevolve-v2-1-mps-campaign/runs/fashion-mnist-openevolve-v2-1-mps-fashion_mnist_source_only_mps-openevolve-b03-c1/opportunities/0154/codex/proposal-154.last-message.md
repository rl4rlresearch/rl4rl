MECHANISM: Centered-view weight boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.044666767120361328125 will preserve 9,210 correct predictions and reduce validation cross-entropy below 0.23114465255737304.

INTENDED_EDIT: Move halfway from the best verified weight toward the larger valid-but-worse probe, preserving all training and arithmetic probability-pooling behavior.

EVIDENCE: Weight 3.044666290283203125 achieved the best score, while 3.04466724395751953125 preserved 9,210 correct but increased cross-entropy; their midpoint is the most informative probe of the local optimum.

<<<<<<< SEARCH
        view_weights = (3.044666290283203125, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.044666767120361328125, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE