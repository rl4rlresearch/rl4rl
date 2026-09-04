MECHANISM: Bottleneck-funded exact parameter EMA

HYPOTHESIS: Replacing the expensive 112-channel residual convolution with a nonlinear 48-channel bottleneck will provide enough runtime headroom for the proven 0.995 EMA to finish and exceed 9,214 correct predictions while retaining 249,854 learned parameters.

INTENDED_EDIT: Preserve batch-32 flip-paired matched-view training, replace the residual branch with a cheaper bottleneck, widen the classifier to 144 units, maintain per-step EMA parameters for final evaluation, and batch each mirrored inference pair.

EVIDENCE: The current full-width design completed with 9,214 correct, while exact 0.995 EMA reached 9,241 but timed out; modest width reductions also timed out, motivating a larger targeted reduction in the dominant residual convolution rather than weakening the established training rule.

<<<<<<< SEARCH
        self.residual = nn.Sequential(
            nn.Conv2d(112, 112, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(112),
        )
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(112 * 3 * 3, 64),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(64, 10),
        )
=======
        self.residual = nn.Sequential(
            nn.Conv2d(112, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 112, kernel_size=1, bias=False),
            nn.BatchNorm2d(112),
        )
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(112 * 3 * 3, 144),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(144, 10),
        )
>>>>>>> REPLACE

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
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    with torch.no_grad():
        torch._foreach_lerp_(optimizer._ema_params, parameters, 0.005)
        if step + 1 == total_steps:
            torch._foreach_copy_(parameters, optimizer._ema_params)

    progress = min(max((step + 1) / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE