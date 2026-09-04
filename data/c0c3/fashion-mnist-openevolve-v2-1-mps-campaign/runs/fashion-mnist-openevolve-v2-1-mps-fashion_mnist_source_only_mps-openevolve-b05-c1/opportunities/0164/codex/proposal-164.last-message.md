MECHANISM: Efficient post-EMA BatchNorm recalibration

HYPOTHESIS: Recomputing BatchNorm statistics from the final 1,536 augmented examples after installing EMA parameters will exceed 9,360 correct predictions by reducing weight–normalization mismatch without the overhead of the timed-out 6,000-example attempt.

INTENDED_EDIT: Cache only the final 16 training batches and use them to reset and cumulatively recompute all BatchNorm running statistics under the final EMA parameters.

EVIDENCE: Averaging BatchNorm buffers reduced validation_correct to 9,351, whereas unaveraged buffers reached 9,360; the more principled post-EMA recalibration attempt timed out, so a substantially cheaper retry remains informative.

<<<<<<< SEARCH
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
        self.bn_recalibration_images: list[torch.Tensor] = []
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer.ema_buffer_pairs = None
    return optimizer
=======
    optimizer.ema_buffer_pairs = None
    optimizer.recalibration_model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels)
=======
) -> torch.Tensor:
    if step + 16 >= total_steps:
        model.bn_recalibration_images.append(images.detach().clone())
    return F.cross_entropy(model(images), labels)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if step + 1 >= total_steps:
            for parameter, average in optimizer.ema_pairs:
                parameter.copy_(average)

    progress = min((step + 1) / max(total_steps, 1), 1.0)
=======
        if step + 1 >= total_steps:
            for parameter, average in optimizer.ema_pairs:
                parameter.copy_(average)

            recalibration_model = optimizer.recalibration_model
            normalization_layers = [
                module
                for module in recalibration_model.modules()
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
            ]
            previous_momenta = [
                module.momentum for module in normalization_layers
            ]
            for module in normalization_layers:
                module.reset_running_stats()
                module.momentum = None

            previous_training = recalibration_model.training
            recalibration_model.train()
            for recalibration_images in (
                recalibration_model.bn_recalibration_images
            ):
                recalibration_model(recalibration_images)

            for module, momentum in zip(
                normalization_layers, previous_momenta
            ):
                module.momentum = momentum
            recalibration_model.train(previous_training)
            recalibration_model.bn_recalibration_images.clear()

    progress = min((step + 1) / max(total_steps, 1), 1.0)
>>>>>>> REPLACE