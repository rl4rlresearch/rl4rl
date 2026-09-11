MECHANISM: Fused parameter EMA with batched flip inference

HYPOTHESIS: A fused 0.995-decay EMA will retain the observed 9,241-correct benefit while batching each flip pair will recover enough verification time to finish and exceed 9,214 correct predictions.

INTENDED_EDIT: Maintain an exact parameter EMA using fused foreach operations, install it after the final update, and evaluate each spatial view with its mirror in one forward pass.

EVIDENCE: The prior EMA attempt reached 9,241 correct versus the current 9,214 but timed out at 82.89 training seconds; fused updates and halving inference forward-call count target that computational failure without changing its successful learning rule.

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
            paired_views = torch.cat((view, view.flip(-1)), dim=0)
            paired_log_probs = F.log_softmax(
                self._forward_once(paired_views), dim=1
            )
            log_prob_views.extend(paired_log_probs.chunk(2, dim=0))
        stacked = torch.stack(log_prob_views, dim=0)
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
    params = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer._ema_params = [
        parameter.detach().clone() for parameter in params
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(max((step + 1) / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    params = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    torch._foreach_lerp_(optimizer._ema_params, params, 0.005)
    if step + 1 == total_steps:
        torch._foreach_copy_(params, optimizer._ema_params)

    progress = min(max((step + 1) / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE