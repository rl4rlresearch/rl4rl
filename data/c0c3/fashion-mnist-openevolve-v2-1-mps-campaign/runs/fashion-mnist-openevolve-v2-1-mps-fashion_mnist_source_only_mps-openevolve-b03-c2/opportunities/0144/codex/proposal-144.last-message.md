MECHANISM: Binary refinement of the upper center-weight boundary

HYPOTHESIS: A center-view weight of 3.39617919921875 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.

INTENDED_EDIT: Move halfway between the accuracy-preserving 3.39599609375 weight and the accuracy-losing 3.3963623046875 weight, update ensemble normalization, and use the verified 1.35605 calibration.

EVIDENCE: Weight 3.39599609375 achieved 9,257 correct at 0.2144189220 cross-entropy, whereas 3.3963623046875 achieved 9,256 correct at 0.2144188389; their midpoint is the most informative remaining probe of the upper 9,257-correct boundary.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.39617919921875 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.35641 * aggregate_logits
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.7923583984375)
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE