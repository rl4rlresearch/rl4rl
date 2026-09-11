MECHANISM: Final-quarter cumulative BatchNorm recalibration

HYPOTHESIS: Rebuilding BatchNorm running statistics over the low-learning-rate final quarter will exceed 9,322 correct predictions while retaining the completed hard-maximum architecture’s runtime.

INTENDED_EDIT: Reset BatchNorm statistics at 75% training progress, then accumulate sample-weighted running statistics over the remaining examples.

EVIDENCE: Sample-weighting only the two tail batches left the 9,320-correct result unchanged versus ordinary BatchNorm, indicating tail influence was not the main issue; the fast hard-maximum model leaves room to test a broader, parameter-free reduction in evaluation-statistic noise.

<<<<<<< SEARCH
    del step, total_steps
    batch_fraction = images.shape[0] / BATCH_SIZE
    batch_norm_momentum = 1.0 - (1.0 - 0.1) ** batch_fraction
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = batch_norm_momentum
=======
    recalibration_start = 3 * total_steps // 4
    batch_examples = images.shape[0]
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if step == recalibration_start:
                module.reset_running_stats()
                module.recalibration_examples = 0
            if step >= recalibration_start:
                seen_examples = getattr(module, "recalibration_examples", 0)
                module.momentum = batch_examples / (
                    seen_examples + batch_examples
                )
                module.recalibration_examples = seen_examples + batch_examples
            else:
                module.momentum = 0.1
>>>>>>> REPLACE