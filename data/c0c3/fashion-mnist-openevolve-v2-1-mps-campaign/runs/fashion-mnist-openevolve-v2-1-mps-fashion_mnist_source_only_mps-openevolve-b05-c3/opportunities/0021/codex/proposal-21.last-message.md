MECHANISM: Center-weighted cardinal test-time augmentation

HYPOTHESIS: Giving the centered views twice the weight of each shifted view will exceed 9,167 correct predictions by retaining the demonstrated benefit of cardinal-view diversity while reducing dependence on displaced predictions.

INTENDED_EDIT: Preserve the best validated training configuration and change the ten-view probability ensemble so each centered orientation has weight two, each shifted orientation has weight one, and the weighted mean uses total weight twelve.

EVIDENCE: Cardinal ensembling improved flip-only evaluation from 9,110 to 9,125 correct, establishing that shifted views add useful evidence; center weighting directly tests whether their benefit comes from complementary predictions rather than requiring the current 80% aggregate weight on shifted images.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits, dim=1)
        probability_sum.add_(
            F.softmax(self._forward_once(images.flip(-1)), dim=1)
        )
=======
        probability_sum = 2.0 * F.softmax(logits, dim=1)
        probability_sum.add_(
            2.0 * F.softmax(self._forward_once(images.flip(-1)), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
        return (probability_sum / 12.0).clamp_min(1e-8).log()
>>>>>>> REPLACE