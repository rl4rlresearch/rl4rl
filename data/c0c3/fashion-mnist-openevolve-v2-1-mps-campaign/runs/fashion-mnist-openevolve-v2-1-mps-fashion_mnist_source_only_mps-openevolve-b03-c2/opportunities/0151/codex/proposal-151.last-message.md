MECHANISM: Binary refinement of the upper center-weight boundary

HYPOTHESIS: A center-view weight of 3.3960161209106445 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441891708374022.

INTENDED_EDIT: Move halfway between the accuracy-preserving 3.3960132598876953 weight and the accuracy-losing 3.39601898193359375 weight, updating ensemble normalization exactly.

EVIDENCE: Weight 3.3960132598876953 achieved 9,257 correct at 0.2144189171 cross-entropy, while 3.39601898193359375 achieved 9,256 correct at the slightly lower 0.2144189125; their midpoint is the most informative unresolved probe.

<<<<<<< SEARCH
            log_weight = math.log(3.3963623046875 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.3960161209106445 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792724609375)
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
>>>>>>> REPLACE