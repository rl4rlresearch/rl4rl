MECHANISM: Exponential-kernel checkpoint weight averaging

HYPOTHESIS: A five-checkpoint approximation of the accuracy-improving 0.995 EMA, combined with pair-batched inference, will finish within the time limit and exceed 9,214 correct predictions.

INTENDED_EDIT: Accumulate five strategically weighted late-training parameter snapshots, install their EMA-kernel approximation after the final update, and evaluate each original/mirrored view pair in one forward pass.

EVIDENCE: Exact 0.995 EMA reached 9,241 correct but timed out from continuous averaging overhead; five checkpoint operations approximate the same temporal weighting at negligible training cost, while Reference Design 3 established pair-batched mirrored inference as valid.

<<<<<<< SEARCH
        log_prob_views = []
        for view in views:
            log_prob_views.append(
                F.log_softmax(self._forward_once(view), dim=1)
            )
            log_prob_views.append(
                F.log_softmax(self._forward_once(view.flip(-1)), dim=1)
            )
        stacked = torch.stack(log_prob_views, dim=0)
=======
        log_prob_views = []
        for view in views:
            paired = torch.cat((view, view.flip(-1)), dim=0)
            paired_log_probs = F.log_softmax(
                self._forward_once(paired),
                dim=1,
            )
            original, mirrored = paired_log_probs.chunk(2, dim=0)
            log_prob_views.extend((original, mirrored))

        stacked = torch.stack(log_prob_views, dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    remaining = total_steps - (step + 1)
    if remaining in (640, 320, 160, 64, 0):
        decay = 0.995
        if remaining == 640:
            weight = decay ** 480
        elif remaining == 320:
            weight = decay ** 240 - decay ** 480
        elif remaining == 160:
            weight = decay ** 112 - decay ** 240
        elif remaining == 64:
            weight = decay ** 32 - decay ** 112
        else:
            weight = 1.0 - decay ** 32

        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_checkpoint_average"):
                optimizer._checkpoint_average = [
                    parameter.detach().clone().mul_(weight)
                    for parameter in parameters
                ]
            else:
                torch._foreach_add_(
                    optimizer._checkpoint_average,
                    parameters,
                    alpha=weight,
                )

            if remaining == 0:
                torch._foreach_copy_(
                    parameters,
                    optimizer._checkpoint_average,
                )
                del optimizer._checkpoint_average
>>>>>>> REPLACE