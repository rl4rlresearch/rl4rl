MECHANISM: Centered-view weight one-ULP refinement

HYPOTHESIS: Raising the centered-view weight by one float32 ULP to 3.0446665287017822265625 will preserve 9,210 correct predictions and reduce cross-entropy below 0.23114465255737304.

INTENDED_EDIT: Probe the only representable float32 weight between the best verified setting and the larger valid-but-worse setting, preserving all other behavior.

EVIDENCE: Weight 3.044666290283203125 is the best verified result, while 3.044666767120361328125 preserved correctness but worsened cross-entropy; the intervening float32 value is the most informative local-optimum probe.

<<<<<<< SEARCH
        view_weights = (3.044666290283203125, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.0446665287017822265625, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE