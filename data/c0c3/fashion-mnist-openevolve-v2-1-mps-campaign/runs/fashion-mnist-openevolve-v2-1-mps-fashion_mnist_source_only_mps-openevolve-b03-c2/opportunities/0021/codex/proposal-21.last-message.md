MECHANISM: Fully batched ten-view inference funding exact parameter EMA

HYPOTHESIS: Processing all ten validation views in one forward call will provide enough runtime headroom for the previously observed 9,241-correct 0.995 EMA to finish verification and exceed 9,214 correct predictions.

INTENDED_EDIT: Preserve the proven training rule, maintain a fused per-step EMA of learned parameters, install it after the final update, and concatenate all spatial and mirrored inference views into one batch.

EVIDENCE: Exact EMA reached 9,241 correct versus the qualified 9,214 but timed out at 82.89 seconds; batching only each flip pair still timed out, motivating the stronger reduction from five inference calls to one.

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
        return torch.logsumexp(stacked, dim=0) - math.log(len(log_prob_views))
=======
        augmented = torch.cat(
            views + tuple(view.flip(-1) for view in views),
            dim=0,
        )
        stacked = F.log_softmax(
            self._forward_once(augmented), dim=1
        ).reshape(len(views) * 2, images.shape[0], 10)
        return torch.logsumexp(stacked, dim=0) - math.log(len(views) * 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
    )
    optimizer._ema_params = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer._ema_values = [
        parameter.detach().clone() for parameter in optimizer._ema_params
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    with torch.no_grad():
        torch._foreach_lerp_(
            optimizer._ema_values,
            optimizer._ema_params,
            0.005,
        )
        if step + 1 >= total_steps:
            torch._foreach_copy_(
                optimizer._ema_params,
                optimizer._ema_values,
            )
>>>>>>> REPLACE