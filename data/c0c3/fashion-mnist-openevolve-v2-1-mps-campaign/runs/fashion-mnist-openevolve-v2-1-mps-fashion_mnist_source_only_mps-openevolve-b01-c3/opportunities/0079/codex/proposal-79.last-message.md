MECHANISM: Decision-invariant fine-grained temperature sharpening

HYPOTHESIS: Scaling the linear-recency ten-view logits by 1.125 will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2096186.

INTENDED_EDIT: Restore the verified linear-recency tail average and apply a conservative evaluation-only scale between the successful 1.10 setting and the unverified 1.15 setting.

EVIDENCE: Linear recency with 1.10 scaling achieved 9,287 correct and reduced cross-entropy from 0.2173399 to 0.2096186; positive scaling cannot change predicted classes, and the 1.15 attempt timed out without subject-level evidence against further sharpening.

<<<<<<< SEARCH
        return view_logits.mean(dim=(0, 1))
=======
        return 1.125 * view_logits.mean(dim=(0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        recency_power = 0.75
=======
        recency_power = 1.0
>>>>>>> REPLACE