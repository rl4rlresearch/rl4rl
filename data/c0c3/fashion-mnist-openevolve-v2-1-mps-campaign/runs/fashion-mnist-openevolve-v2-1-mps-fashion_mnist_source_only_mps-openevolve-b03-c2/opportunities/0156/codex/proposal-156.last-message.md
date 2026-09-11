MECHANISM: Single-ULP local ensemble-weight refinement

HYPOTHESIS: The next float32 center-view weight below 3.3960161209106445 will preserve 9,257 correct predictions and reduce cross-entropy below 0.21441891098022461.

INTENDED_EDIT: Set the center-view weight to 3.3960158824920654 and update its exact ensemble normalization.

EVIDENCE: Weight 3.3960161209106445 has the best verified 9,257-correct cross-entropy, while its immediate higher float32 neighbor worsened cross-entropy; the immediate lower neighbor is the most informative unresolved local probe.

<<<<<<< SEARCH
            log_weight = math.log(3.39617919921875 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.3960158824920654 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.7923583984375)
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.79203176498413)
>>>>>>> REPLACE