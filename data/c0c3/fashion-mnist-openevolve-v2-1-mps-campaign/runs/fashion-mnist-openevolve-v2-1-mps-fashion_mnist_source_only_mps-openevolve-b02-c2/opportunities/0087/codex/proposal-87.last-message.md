MECHANISM: Top-four saliency with sampled late-state EMA

HYPOTHESIS: Restoring top-four channel saliency and evaluating a sparsely sampled EMA of the final training half will exceed 9,322 correct predictions by combining the best verified attention descriptor with lower low-learning-rate checkpoint variance.

INTENDED_EDIT: Restore exact top-four attention, ordinary BatchNorm momentum, and maintain a low-overhead EMA of parameters and floating BatchNorm state every 16 steps during the final half of training.

EVIDENCE: Top-four saliency produced the best verified count of 9,322; sample-weighted BatchNorm remained at 9,320 while increasing runtime, so ordinary BatchNorm provides headroom for sparse late-state averaging.

<<<<<<< SEARCH
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        averaged_state = getattr(self, "_averaged_state", None)
        if not self.training and averaged_state is not None:
            with torch.no_grad():
                current_state = self.state_dict()
                for name, average in averaged_state.items():
                    current_state[name].copy_(average)
            self._averaged_state = None

        logits = self._forward_once(images)
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
    optimizer._averaging_model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    batch_fraction = images.shape[0] / BATCH_SIZE
    batch_norm_momentum = 1.0 - (1.0 - 0.1) ** batch_fraction
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = batch_norm_momentum

    paired_images = torch.cat(
=======
    del step, total_steps
    paired_images = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    progress = min(step / max(total_steps, 1), 1.0)
    if progress >= 0.5 and step % 16 == 0:
        model = optimizer._averaging_model
        current_state = {
            name: tensor
            for name, tensor in model.state_dict().items()
            if tensor.is_floating_point()
        }
        averaged_state = getattr(model, "_averaged_state", None)
        if averaged_state is None:
            model._averaged_state = {
                name: tensor.detach().clone()
                for name, tensor in current_state.items()
            }
        else:
            with torch.no_grad():
                for name, average in averaged_state.items():
                    average.lerp_(current_state[name].detach(), 0.2)

    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE