MECHANISM: Exponential trajectory weight averaging

HYPOTHESIS: Applying a 0.99-decay EMA to the 9,229-correct differential-basis, flip-ensemble-supervised model will exceed 9,229 correct predictions by reducing sensitivity to the final noisy AdamW iterate.

INTENDED_EDIT: Restore the strongest differential-image-basis and flip-ensemble loss, then maintain an EMA of learned parameters and floating-point BatchNorm buffers and install it after the final optimizer step.

EVIDENCE: Reference Design 3 reached 9,229 correct after aligning training with flip-averaged evaluation; its cosine schedule retains a nonzero learning-rate floor, motivating trajectory averaging as an isolated way to stabilize the final classifier without adding learned parameters.

<<<<<<< SEARCH
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
=======
    @staticmethod
    def _image_basis(images: torch.Tensor) -> torch.Tensor:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        local_mean = F.avg_pool2d(padded, kernel_size=3, stride=1)
        local_contrast = images - local_mean

        gradient_x = 0.125 * (
            padded[:, :, :-2, 2:]
            + 2.0 * padded[:, :, 1:-1, 2:]
            + padded[:, :, 2:, 2:]
            - padded[:, :, :-2, :-2]
            - 2.0 * padded[:, :, 1:-1, :-2]
            - padded[:, :, 2:, :-2]
        )
        gradient_y = 0.125 * (
            padded[:, :, 2:, :-2]
            + 2.0 * padded[:, :, 2:, 1:-1]
            + padded[:, :, 2:, 2:]
            - padded[:, :, :-2, :-2]
            - 2.0 * padded[:, :, :-2, 1:-1]
            - padded[:, :, :-2, 2:]
        )
        edge_energy = torch.sqrt(
            gradient_x.square() + gradient_y.square() + 1.0e-6
        )
        return torch.cat(
            (
                images,
                local_contrast,
                gradient_x,
                gradient_y,
                edge_energy,
            ),
            dim=1,
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
    )
    optimizer._ema_parameters = None
    optimizer._ema_buffers = [
        buffer for buffer in model.buffers() if buffer.is_floating_point()
    ]
    optimizer._ema_buffer_values = None
    optimizer._ema_updates = 0
    optimizer._ema_total_steps = total_steps
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    flip_mask = torch.rand(images.size(0), device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None], images.flip(-1), images
    )
    return images, labels
=======
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
=======
    del step, total_steps
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)

    individual_loss = F.cross_entropy(
        paired_logits, paired_labels, label_smoothing=0.03
    )
    ensemble_loss = F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=0.03,
    )
    return 0.5 * individual_loss + 0.5 * ensemble_loss
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer._ema_updates += 1

    with torch.no_grad():
        if optimizer._ema_parameters is None:
            optimizer._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
            optimizer._ema_buffer_values = [
                buffer.detach().clone() for buffer in optimizer._ema_buffers
            ]
        else:
            for average, parameter in zip(
                optimizer._ema_parameters, parameters
            ):
                average.lerp_(parameter.detach(), 0.01)
            for average, buffer in zip(
                optimizer._ema_buffer_values, optimizer._ema_buffers
            ):
                average.lerp_(buffer.detach(), 0.01)

        if optimizer._ema_updates >= optimizer._ema_total_steps:
            for parameter, average in zip(
                parameters, optimizer._ema_parameters
            ):
                parameter.copy_(average)
            for buffer, average in zip(
                optimizer._ema_buffers, optimizer._ema_buffer_values
            ):
                buffer.copy_(average)
>>>>>>> REPLACE