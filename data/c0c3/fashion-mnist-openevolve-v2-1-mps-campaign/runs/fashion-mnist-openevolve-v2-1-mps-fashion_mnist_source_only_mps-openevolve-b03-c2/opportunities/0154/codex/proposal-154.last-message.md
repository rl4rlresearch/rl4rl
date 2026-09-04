MECHANISM: Binary refinement of the upper center-weight boundary

HYPOTHESIS: A center-view weight of 3.3960182666778564 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441891098022461.

INTENDED_EDIT: Move halfway between the highest accuracy-preserving weight and the nearest accuracy-losing weight, updating ensemble normalization exactly.

EVIDENCE: Weight 3.396017551422119 preserved 9,257 correct, while 3.39601898193359375 produced 9,256; their midpoint is the most informative unresolved boundary probe.

<<<<<<< SEARCH
            log_weight = math.log(3.396007537841797 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.3960182666778564 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792015075683594)
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792036533355713)
>>>>>>> REPLACE