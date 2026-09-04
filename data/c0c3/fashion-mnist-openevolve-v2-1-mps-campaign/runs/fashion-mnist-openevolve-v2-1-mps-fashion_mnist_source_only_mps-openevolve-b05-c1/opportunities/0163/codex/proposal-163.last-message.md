MECHANISM: Post-EMA BatchNorm recalibration

HYPOTHESIS: Recomputing BatchNorm statistics under the final EMA parameters using the latest 6,000 training examples will exceed 9,360 correct predictions by correcting the weight–normalization mismatch more faithfully than averaging stale running statistics.

INTENDED_EDIT: Cache the latest 64 augmented batches, then reset and cumulatively recompute all BatchNorm running statistics after installing the final EMA parameters.

EVIDENCE: Averaging BatchNorm buffers reduced validation_correct to 9,351, while leaving them unaveraged reached 9,360; this suggests normalization state matters, but averaged statistics are not valid statistics for the averaged network.

<<<<<<< SEARCH
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
        self.bn_recalibration_batches: list[torch.Tensor] = []
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer.ema_buffer_pairs = None
    return optimizer
=======
    optimizer.ema_buffer_pairs = None
    optimizer.model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels)
=======
    del step, total_steps
    model.bn_recalibration_batches.append(images.detach())
    if len(model.bn_recalibration_batches) > 64:
        model.bn_recalibration_batches.pop(0)
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

            model = optimizer.model
            batch_norms = [
                module
                for module in model.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
            ]
            original_momenta = [
                module.momentum for module in batch_norms
            ]
            was_training = model.training
            model.train()
            for module in batch_norms:
                module.reset_running_stats()
                module.momentum = None
            for images in model.bn_recalibration_batches:
                model(images)
            for module, momentum in zip(
                batch_norms, original_momenta
            ):
                module.momentum = momentum
            model.train(was_training)
            model.bn_recalibration_batches.clear()

    progress = min((step + 1) / max(total_steps, 1), 1.0)
>>>>>>> REPLACE