MECHANISM: Two-view inference funded sparse parameter EMA

HYPOTHESIS: Sampling a 0.995-equivalent EMA every 16 updates and replacing ten-view inference with center-plus-mirror inference will finish within the time limit and exceed 9,214 correct predictions.

INTENDED_EDIT: Maintain a low-overhead EMA of learned parameters, install it after the final update, and evaluate only the batched centered image and its mirror.

EVIDENCE: Exact 0.995 EMA reached 9,241 correct, 27 above the current 9,214, while shifted ten-view inference improved Reference Design 3 by only seven correct; removing eight inference views should fund EMA with a smaller expected accuracy cost.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
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
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        paired = torch.cat((images, images.flip(-1)), dim=0)
        paired_log_probs = F.log_softmax(self._forward_once(paired), dim=1)
        original, mirrored = paired_log_probs.chunk(2, dim=0)
        stacked = torch.stack((original, mirrored), dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(2.0)
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
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer._ema_last_step = 0
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
    completed_steps = step + 1
    is_final_step = completed_steps >= total_steps
    if completed_steps % 16 == 0 or is_final_step:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        elapsed = completed_steps - optimizer._ema_last_step
        ema_weight = 1.0 - 0.995 ** elapsed
        with torch.no_grad():
            torch._foreach_lerp_(
                optimizer._ema_params,
                parameters,
                ema_weight,
            )
            if is_final_step:
                torch._foreach_copy_(parameters, optimizer._ema_params)
        optimizer._ema_last_step = completed_steps

    progress = min(max(completed_steps / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE