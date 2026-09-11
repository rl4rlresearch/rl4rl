MECHANISM: Argmax-preserving ensemble temperature sharpening

HYPOTHESIS: Restoring the verified 3.25:1 center weighting will recover 9,256 correct predictions, while scaling its aggregate logits by 1.03 will preserve those argmaxes and reduce cross-entropy below 0.2347709.

INTENDED_EDIT: Restore the best center-view weight and apply mild post-ensemble logit sharpening.

EVIDENCE: The 3.25:1 design achieved the best verified correctness at 9,256; the previous 1.03 sharpening attempt only timed out, providing no contrary accuracy or cross-entropy evidence, and positive scaling cannot alter its predicted classes.

<<<<<<< SEARCH
            log_weight = math.log(3.75 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(15.5)
=======
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.03 * aggregate_logits
>>>>>>> REPLACE