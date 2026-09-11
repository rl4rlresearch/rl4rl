MECHANISM: Fine-grained label-smoothing interpolation

HYPOTHESIS: Increasing label smoothing slightly to 0.021 will retain or exceed 9,256 correct predictions while lowering validation cross-entropy below 0.2144522301.

INTENDED_EDIT: Change only the label-smoothing coefficient from 0.02 to 0.021, preserving the verified architecture, augmentation, optimizer, ensemble, and calibration.

EVIDENCE: Label smoothing 0.025 achieved substantially lower cross-entropy but lost three correct predictions, while 0.01 lost seventeen; a minimal upward step tests whether some calibration benefit is available before accuracy degrades.

<<<<<<< SEARCH
        label_smoothing=0.02,
=======
        label_smoothing=0.021,
>>>>>>> REPLACE