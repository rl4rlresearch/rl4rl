MECHANISM: Geometric ten-view evidence pooling

HYPOTHESIS: Averaging logits across all ten translation/flip views will exceed 9,286 correct predictions by rewarding class evidence that remains consistent across transformations.

INTENDED_EDIT: Replace probability-space mixing across translations with a single uniform logit average across all ten evaluation views; training and parameter averaging remain unchanged.

EVIDENCE: The current endpoint-reconstructed tail average is strongest at 9,286 correct, while uniform probability pooling over all ten views timed out; consistent logit pooling is the complementary computation-neutral aggregation test and removes the current asymmetric treatment of flips and translations.

<<<<<<< SEARCH
        position_logits = view_logits.mean(dim=1)
        return torch.logsumexp(
            F.log_softmax(position_logits, dim=-1), dim=0
        ) - math.log(position_logits.shape[0])
=======
        return view_logits.mean(dim=(0, 1))
>>>>>>> REPLACE