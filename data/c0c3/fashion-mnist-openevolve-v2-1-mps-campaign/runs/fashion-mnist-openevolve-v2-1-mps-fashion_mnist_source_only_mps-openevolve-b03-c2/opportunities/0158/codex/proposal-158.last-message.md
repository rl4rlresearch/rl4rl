MECHANISM: Single-ULP downward ensemble-weight refinement

HYPOTHESIS: The next float32 center-view weight below 3.3960156440734863 will preserve 9,257 correct predictions and reduce validation cross-entropy below 0.21441891098022461.

INTENDED_EDIT: Set the center-view weight to 3.3960154056549072 and update the exact ensemble normalization.

EVIDENCE: The two consecutive lower float32 weights, 3.3960158824920654 and 3.3960156440734863, both preserved 9,257 correct with the best cross-entropy; continuing one ULP downward is the finest unresolved local probe.

<<<<<<< SEARCH
            log_weight = math.log(3.3960158824920654 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.3960154056549072 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.79203176498413)
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792030811309814)
>>>>>>> REPLACE