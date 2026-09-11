MECHANISM: Late-training exponential weight averaging with prediction ensembling

HYPOTHESIS: Ensembling the final model with a short-horizon EMA of its second-half trajectory will exceed 9,167 correct predictions by reducing augmentation and optimization variance.

INTENDED_EDIT: Track an EMA of parameters and floating-point BatchNorm buffers during the second half of training, then average its ten-view probabilities with the final model’s ten-view probabilities.

EVIDENCE: Translation-flip probability ensembling improved validation correct from 9,138 to 9,167, showing that averaging complementary predictions benefits this model; temporal weight ensembling extends that successful variance-reduction mechanism without changing training exposure or learned-parameter count.

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
        log_probabilities = []
        for view in views:
            log_probabilities.append(
                F.log_softmax(self._forward_once(view), dim=1)
            )
            log_probabilities.append(
                F.log_softmax(self._forward_once(view.flip(-1)), dim=1)
            )
        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
=======
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

        def collect_log_probabilities() -> list[torch.Tensor]:
            outputs = []
            for view in views:
                outputs.append(
                    F.log_softmax(self._forward_once(view), dim=1)
                )
                outputs.append(
                    F.log_softmax(self._forward_once(view.flip(-1)), dim=1)
                )
            return outputs

        log_probabilities = collect_log_probabilities()
        ema_state = getattr(self, "_ema_state", None)
        if ema_state is not None:
            live_state = dict(self.named_parameters())
            live_state.update(dict(self.named_buffers()))
            backups = {}
            with torch.no_grad():
                for name, average in ema_state.items():
                    tensor = live_state[name]
                    backups[name] = tensor.detach().clone()
                    tensor.copy_(average)
            try:
                log_probabilities.extend(collect_log_probabilities())
            finally:
                with torch.no_grad():
                    for name, backup in backups.items():
                        live_state[name].copy_(backup)

        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
>>>>>>> REPLACE

<<<<<<< SEARCH
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=4e-4,
    )
=======
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=4e-4,
    )
    optimizer._ema_model = model
    optimizer._ema_state = None
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> None:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    warmup_fraction = 0.08
=======
) -> None:
    ema_model = getattr(optimizer, "_ema_model", None)
    if ema_model is not None and step >= total_steps // 2:
        live_state = dict(ema_model.named_parameters())
        live_state.update(dict(ema_model.named_buffers()))
        floating_state = {
            name: tensor
            for name, tensor in live_state.items()
            if tensor.is_floating_point()
        }
        if optimizer._ema_state is None:
            optimizer._ema_state = {
                name: tensor.detach().clone()
                for name, tensor in floating_state.items()
            }
        else:
            decay = 0.98
            with torch.no_grad():
                for name, average in optimizer._ema_state.items():
                    average.mul_(decay).add_(
                        floating_state[name].detach(), alpha=1.0 - decay
                    )
        ema_model._ema_state = optimizer._ema_state

    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    warmup_fraction = 0.08
>>>>>>> REPLACE