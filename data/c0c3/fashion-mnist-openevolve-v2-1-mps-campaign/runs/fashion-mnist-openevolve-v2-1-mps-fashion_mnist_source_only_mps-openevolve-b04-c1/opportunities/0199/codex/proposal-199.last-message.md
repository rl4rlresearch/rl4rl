MECHANISM: Late BatchNorm statistic stabilization

HYPOTHESIS: Gradually reducing BatchNorm momentum from 0.10 to 0.05 during the final half of training will exceed 9,330 correct predictions by suppressing terminal minibatch noise while avoiding the excessive historical lag that hurt full EMA averaging of BatchNorm buffers.

INTENDED_EDIT: Preserve default BatchNorm updates through half of training, then linearly lower all BatchNorm momentum values to 0.05 by the final step.

EVIDENCE: Keeping final BatchNorm buffers produced 9,330 correct predictions, while EMA-averaging them fell to 9,327; this motivates a moderate late-stage smoothing of the successful live statistics rather than full parameter-rate averaging.

<<<<<<< SEARCH
    progress = min(step / max(total_steps, 1), 1.0)
    dropout_decay = max(progress - 0.50, 0.0) / 0.50
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
=======
    progress = min(step / max(total_steps, 1), 1.0)
    dropout_decay = max(progress - 0.50, 0.0) / 0.50
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    batch_norm_momentum = 0.10 - 0.05 * dropout_decay
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = batch_norm_momentum
    logits = model(images)
>>>>>>> REPLACE