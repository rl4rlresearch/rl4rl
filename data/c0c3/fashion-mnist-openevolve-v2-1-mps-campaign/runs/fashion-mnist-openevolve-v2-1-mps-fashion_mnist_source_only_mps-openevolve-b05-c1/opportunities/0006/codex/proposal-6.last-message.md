MECHANISM: Probability-space cardinal-view ensembling

HYPOTHESIS: Arithmetic probability averaging across the existing ten views will exceed 9,159 correct predictions by reducing the disproportionate effect of a poorly aligned view on the true class.

INTENDED_EDIT: Replace logit averaging with numerically stable averaging of per-view class probabilities while leaving training, views, parameters, and runtime unchanged.

EVIDENCE: Cardinal views improved correctness from 9,134 to 9,159, while adding diagonal views reduced it to 9,153, indicating heterogeneous view reliability; probability averaging is less punitive than the current geometric pooling when one translation strongly disagrees.

<<<<<<< SEARCH
        logits = self._forward_once(views[0])
        logits = logits + self._forward_once(views[0].flip(-1))
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        return logits / (2.0 * len(views))
=======
        log_probability_sum = F.log_softmax(
            self._forward_once(views[0]), dim=1
        )
        log_probability_sum = torch.logaddexp(
            log_probability_sum,
            F.log_softmax(self._forward_once(views[0].flip(-1)), dim=1),
        )
        for view in views[1:]:
            log_probability_sum = torch.logaddexp(
                log_probability_sum,
                F.log_softmax(self._forward_once(view), dim=1),
            )
            log_probability_sum = torch.logaddexp(
                log_probability_sum,
                F.log_softmax(self._forward_once(view.flip(-1)), dim=1),
            )
        return log_probability_sum - math.log(2.0 * len(views))
>>>>>>> REPLACE