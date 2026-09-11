MECHANISM: Channels-last acceleration for BatchNorm-aware tail EMA

HYPOTHESIS: Channels-last convolution will offset the observed EMA runtime regression, allowing BatchNorm-buffer-aware tail EMA to finish verification while retaining at least its demonstrated 9,255 correct predictions.

INTENDED_EDIT: Run the unchanged CNN in channels-last memory format and extend the existing tail EMA to average and restore floating-point BatchNorm buffers alongside model parameters.

EVIDENCE: BatchNorm-buffer-aware EMA improved validation_correct from 9,254 to 9,255 but required 84.95 seconds; repeated timeouts make a learning-neutral convolution layout optimization the most direct way to preserve that measured accuracy gain while reducing runtime.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.1
        return logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(memory_format=torch.channels_last)
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.1
        return logits


def build_model() -> nn.Module:
    return ImageClassifier().to(memory_format=torch.channels_last)


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.5e-3, weight_decay=1e-4
    )
    optimizer._model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    if completed_steps >= total_steps // 2:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        ema_parameters = getattr(optimizer, "_ema_parameters", None)
        with torch.no_grad():
            ema_start = total_steps // 2
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._ema_parameters = ema_parameters
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.01)

            if completed_steps >= total_steps:
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.copy_(average)
=======
    if completed_steps >= total_steps // 2:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        buffers = [
            buffer
            for buffer in optimizer._model.buffers()
            if buffer.is_floating_point()
        ]
        ema_parameters = getattr(optimizer, "_ema_parameters", None)
        ema_buffers = getattr(optimizer, "_ema_buffers", None)
        with torch.no_grad():
            ema_start = total_steps // 2
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                ema_buffers = [buffer.detach().clone() for buffer in buffers]
                optimizer._ema_parameters = ema_parameters
                optimizer._ema_buffers = ema_buffers
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.01)
                for average, buffer in zip(ema_buffers, buffers):
                    average.lerp_(buffer.detach(), 0.01)

            if completed_steps >= total_steps:
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.copy_(average)
                for buffer, average in zip(buffers, ema_buffers):
                    buffer.copy_(average)
>>>>>>> REPLACE