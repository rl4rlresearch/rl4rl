MECHANISM: Smaller-batch update densification

HYPOTHESIS: Reducing the batch size from 96 to 80 will exceed 9,345 correct predictions by providing 1,250 optimizer steps instead of 1,042 and modestly noisier gradients within the same fixed 100,000-example exposure.

INTENDED_EDIT: Set the training batch size to 80 while preserving the architecture, optimizer, schedule shape, augmentation, and inference ensemble.

EVIDENCE: Multiple representation additions failed to improve the 9,345-correct result or exceeded the time limit, while every reported run used only 1,042 updates at batch size 96; increasing update density tests whether optimization exposure, rather than missing model capacity, is now the limiting factor.

<<<<<<< SEARCH
BATCH_SIZE = 96
=======
BATCH_SIZE = 80
>>>>>>> REPLACE