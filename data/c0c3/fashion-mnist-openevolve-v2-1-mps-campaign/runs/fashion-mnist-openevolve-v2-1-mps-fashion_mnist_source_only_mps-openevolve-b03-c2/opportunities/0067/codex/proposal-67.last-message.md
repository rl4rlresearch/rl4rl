MECHANISM: Argmax-preserving ensemble temperature sharpening

HYPOTHESIS: Restoring the verified 3.25:1 center weight will recover 9,256 correct predictions, while multiplying the aggregated logits by 1.03 will preserve every argmax and reduce cross-entropy below 0.2347709.

INTENDED_EDIT: Restore the best verified center-view weighting and apply mild post-ensemble sharpening to the returned logits.

EVIDENCE: The 3.25:1 design achieved the best result with 9,256 correct; nearby higher weights reduced cross-entropy but lost predictions. Positive logit scaling leaves those 3.25:1 predictions unchanged while targeting the tie-breaking metric.

<<<<<<< SEARCH
            log_weight = math.log(3.5 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(15.0)
=======
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        ensemble_log_probs = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return ensemble_log_probs * 1.03
>>>>>>> REPLACE