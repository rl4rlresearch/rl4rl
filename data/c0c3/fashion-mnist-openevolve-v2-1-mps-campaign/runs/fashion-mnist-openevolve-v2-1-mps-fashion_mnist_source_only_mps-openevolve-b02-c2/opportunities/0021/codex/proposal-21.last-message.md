MECHANISM: Shared average-plus-top-two channel attention

HYPOTHESIS: Averaging the two strongest spatial activations per channel will exceed 9,322 correct predictions by retaining more peak evidence than top-four averaging while preserving some robustness over the 9,320-correct hard maximum.

INTENDED_EDIT: Change only the salient channel descriptor from a top-four mean to a top-two mean, preserving the best architecture, paired-view training, and flip-ensemble inference.

EVIDENCE: Top-four channel attention achieved 9,322 correct versus 9,320 for hard maxima, while hard maxima had slightly lower cross-entropy; top-two averaging tests the closest computational midpoint without the extra spatial top-k work that timed out.

<<<<<<< SEARCH
        channel_salient = features.flatten(2).topk(4, dim=2).values
=======
        channel_salient = features.flatten(2).topk(2, dim=2).values
>>>>>>> REPLACE