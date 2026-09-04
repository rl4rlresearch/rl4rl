MECHANISM: Exposure-matched test-time view weighting

HYPOTHESIS: Weighting spatial inference views according to their 60/10/10/10/10 training exposure will exceed 9,232 correct predictions by retaining shift robustness while reducing the influence of rarely trained shifted views.

INTENDED_EDIT: Replace uniform ten-view probability averaging with a weighted mixture: 60% total weight for the centered view and 10% for each cardinal shift, split equally between original and mirrored orientations.

EVIDENCE: The current design achieved 9,232 correct, while shifted ten-view inference improved Reference Design 3 by only seven correct. Because training exposes centered images 60% of the time but uniform inference gives them only 20% weight, matching inference weights to the observed training distribution is a compute-neutral refinement of that marginal TTA benefit.

<<<<<<< SEARCH
        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(len(log_prob_views))
=======
        stacked = torch.stack(log_prob_views, dim=0)
        log_weights = stacked.new_tensor(
            (0.30, 0.30, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
        ).log().view(-1, 1, 1)
        return torch.logsumexp(stacked + log_weights, dim=0)
>>>>>>> REPLACE