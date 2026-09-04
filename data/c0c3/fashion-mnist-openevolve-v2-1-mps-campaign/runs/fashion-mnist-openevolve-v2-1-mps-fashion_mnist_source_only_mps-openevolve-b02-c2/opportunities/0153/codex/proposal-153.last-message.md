MECHANISM: Sparse late exponential weight averaging

HYPOTHESIS: Applying a low-overhead EMA over the latter half of the reliable hard-maximum training trajectory will finish within the time limit and exceed 9,322 correct predictions by reducing late optimization noise.

INTENDED_EDIT: Restore hard-maximum channel attention and maintain an EMA every four optimizer steps, installing the averaged parameters before validation.

EVIDENCE: Hard-maximum attention repeatedly finished in 66.6–75.3 seconds at about 9,320 correct, while the 9,322-correct top-four design finished near the limit once and subsequently timed out; this trades two baseline predictions for dependable runtime headroom and a previously untested trajectory-level improvement.

<<<<<<< SEARCH
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
=======
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
    optimizer.averaged_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    optimizer.model_parameters = list(model.parameters())
    optimizer.average_updates = 0
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    if step >= total_steps // 2 and step % 4 == 0:
        with torch.no_grad():
            if optimizer.average_updates == 0:
                for average, parameter in zip(
                    optimizer.averaged_parameters,
                    optimizer.model_parameters,
                ):
                    average.copy_(parameter)
            else:
                torch._foreach_lerp_(
                    optimizer.averaged_parameters,
                    optimizer.model_parameters,
                    0.10,
                )
        optimizer.average_updates += 1

    if step >= total_steps - 1 and optimizer.average_updates:
        with torch.no_grad():
            for parameter, average in zip(
                optimizer.model_parameters,
                optimizer.averaged_parameters,
            ):
                parameter.copy_(average)

    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE