MECHANISM: Decision-invariant temperature sharpening

HYPOTHESIS: Scaling the pooled validation logits by 1.20 will preserve exactly 9,287 correct predictions while reducing validation cross-entropy below 0.2096186.

INTENDED_EDIT: Increase evaluation-only ten-view logit scaling from 1.10 to 1.20 without altering training or predicted classes.

EVIDENCE: Scaling the same linear-recency ten-view logits from 1.00 to 1.10 preserved 9,287 correct predictions and reduced cross-entropy from 0.2173399 to 0.2096186, motivating one further equal sharpening increment.

<<<<<<< SEARCH
        return 1.10 * view_logits.mean(dim=(0, 1))
=======
        return 1.20 * view_logits.mean(dim=(0, 1))
>>>>>>> REPLACE