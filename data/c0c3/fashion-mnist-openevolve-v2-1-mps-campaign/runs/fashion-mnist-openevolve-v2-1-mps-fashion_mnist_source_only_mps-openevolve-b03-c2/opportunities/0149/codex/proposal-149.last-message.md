MECHANISM: Retry unresolved upper-bound refinement

HYPOTHESIS: A center-view weight of 3.396007537841797 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.

INTENDED_EDIT: Retry the previously timed-out boundary probe and update the ensemble normalization constant exactly.

EVIDENCE: Weight 3.39599609375 achieved 9,257 correct, while 3.39601898193359375 achieved 9,256 with lower cross-entropy; the intervening 3.396007537841797 probe timed out and therefore remains the most informative unresolved point.

<<<<<<< SEARCH
            log_weight = math.log(3.39599609375 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.396007537841797 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.7919921875)
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792015075683594)
>>>>>>> REPLACE