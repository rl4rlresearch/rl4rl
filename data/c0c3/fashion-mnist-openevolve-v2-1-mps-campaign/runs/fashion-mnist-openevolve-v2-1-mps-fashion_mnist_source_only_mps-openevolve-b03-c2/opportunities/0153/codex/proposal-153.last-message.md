MECHANISM: Retry unresolved upper-bound refinement

HYPOTHESIS: A center-view weight of 3.396017551422119 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441891098022461.

INTENDED_EDIT: Retry the previously timed-out midpoint probe and update the ensemble normalization exactly.

EVIDENCE: Weight 3.3960161209106445 achieved 9,257 correct, while 3.39601898193359375 achieved 9,256; the intervening 3.396017551422119 probe timed out and remains the most informative unresolved point.

<<<<<<< SEARCH
            log_weight = math.log(3.3960161209106445 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.396017551422119 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792035102844238)
>>>>>>> REPLACE