MECHANISM: Stop-gradient top-four channel saliency

HYPOTHESIS: Detaching only the top-four descriptor from backbone gradients will exceed 9,322 correct predictions by retaining the successful forward gate while preventing sparse selected-location gradients from destabilizing feature learning.

INTENDED_EDIT: Compute the identical top-four activation mean from detached features; preserve all forward behavior, parameters, training loss, and inference ensembling.

EVIDENCE: Shared top-four attention achieved the best result at 9,322 correct, while the dense-gradient log-mean-exp alternative fell to 9,301 and subsequent top-k experiments repeatedly timed out; this isolates salient-descriptor backpropagation while modestly reducing backward work.

<<<<<<< SEARCH
        channel_salient = features.flatten(2).topk(4, dim=2).values
=======
        channel_salient = features.detach().flatten(2).topk(4, dim=2).values
>>>>>>> REPLACE