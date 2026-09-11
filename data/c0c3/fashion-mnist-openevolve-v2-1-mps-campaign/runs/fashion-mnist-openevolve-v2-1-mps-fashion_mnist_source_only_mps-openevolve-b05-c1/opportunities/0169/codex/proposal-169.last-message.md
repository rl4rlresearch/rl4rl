MECHANISM: Fine-grained EMA horizon interpolation

HYPOTHESIS: Reducing the EMA decay cap from 0.98 to 0.979 will retain all 9,360 correct predictions while lowering validation cross-entropy, strictly improving validation_score.

INTENDED_EDIT: Shorten only the late-training parameter-averaging horizon; preserve the architecture, loss, augmentation, schedule, TTA, BatchNorm handling, and calibration.

EVIDENCE: The 0.975 EMA achieved lower cross-entropy but lost three correct predictions, while 0.99 lost eleven and worsened cross-entropy; testing 0.979 isolates a narrow point between the accuracy-leading 0.98 setting and the better-calibrated 0.975 setting.

<<<<<<< SEARCH
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
=======
            ema_decay = min(0.979, (updates + 1.0) / (updates + 10.0))
>>>>>>> REPLACE