MECHANISM: Shorter-horizon late-training parameter EMA

HYPOTHESIS: Reducing the EMA decay cap from 0.98 to 0.975 will exceed 9,360 correct predictions by keeping averaged parameters better aligned with the unaveraged BatchNorm statistics.

INTENDED_EDIT: Change only the parameter EMA decay cap, preserving the verified architecture, augmentation, schedule, TTA, and calibration.

EVIDENCE: Increasing the cap to 0.99 reduced validation_correct from 9,360 to 9,349, while averaging BatchNorm buffers reduced it to 9,351; a modestly shorter parameter-average horizon directly tests the favorable direction without reintroducing buffer averaging.

<<<<<<< SEARCH
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
=======
            ema_decay = min(0.975, (updates + 1.0) / (updates + 10.0))
>>>>>>> REPLACE