MECHANISM: Midpoint center-weight refinement

HYPOTHESIS: A 3.275:1 center-to-shift inference weight will retain the 9,256 correct predictions achieved at 3.25:1 while lowering cross-entropy below 0.2347709.

INTENDED_EDIT: Reduce each center-view weight from 3.30 to 3.275 and normalize the probability mixture by 14.55.

EVIDENCE: The verified 3.25:1 design achieved the best 9,256 correct, while 3.30:1 lost one correct but lowered cross-entropy; their midpoint is the most informative untested tradeoff.

<<<<<<< SEARCH
            log_weight = math.log(3.3 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.275 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return torch.logsumexp(stacked, dim=0) - math.log(14.6)
=======
        return torch.logsumexp(stacked, dim=0) - math.log(14.55)
>>>>>>> REPLACE