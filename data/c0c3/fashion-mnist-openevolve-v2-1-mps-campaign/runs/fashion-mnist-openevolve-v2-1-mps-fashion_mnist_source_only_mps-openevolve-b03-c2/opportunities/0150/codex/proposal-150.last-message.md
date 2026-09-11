MECHANISM: Binary refinement of the upper center-weight boundary

HYPOTHESIS: A center-view weight of 3.3960132598876953 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441891479492187.

INTENDED_EDIT: Move halfway between the accuracy-preserving 3.396007537841797 weight and the accuracy-losing 3.39601898193359375 weight, updating the ensemble normalization exactly.

EVIDENCE: Weight 3.396007537841797 achieved 9,257 correct at 0.2144189148 cross-entropy, while 3.39601898193359375 achieved 9,256 correct at the slightly lower 0.2144189125; their midpoint is the most informative unresolved probe.

<<<<<<< SEARCH
            log_weight = math.log(3.3974609375 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.3960132598876953 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.794921875)
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.79202651977539)
>>>>>>> REPLACE