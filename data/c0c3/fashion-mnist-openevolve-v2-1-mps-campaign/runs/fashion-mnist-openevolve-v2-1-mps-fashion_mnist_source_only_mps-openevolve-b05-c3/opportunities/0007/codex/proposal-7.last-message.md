MECHANISM: Higher-frequency optimization with exposure-matched regularization and EMA

HYPOTHESIS: Reducing batch size from 64 to 32 will exceed 9,128 correct predictions because the previous halving increased correct predictions and reduced cross-entropy, while halving AdamW decay and EMA update rate preserves their behavior per training example.

INTENDED_EDIT: Use batch size 32, halve weight decay to 7.5e-5, and halve EMA interpolation to 0.0025 while retaining the best validated architecture, augmentation, schedule, and ten-view inference.

EVIDENCE: Moving from batch 128 to 64 increased validation correct from 9,125 to 9,128 and reduced cross-entropy from 0.24994 to 0.24579, directly supporting another controlled increase in optimizer-update frequency.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 2.0
BASE_LR = 3.0e-3
=======
BATCH_SIZE = 32
GRAD_CLIP_NORM = 2.0
BASE_LR = 3.0e-3
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=BASE_LR * 0.2,
        weight_decay=1.5e-4,
=======
        lr=BASE_LR * 0.2,
        weight_decay=7.5e-5,
>>>>>>> REPLACE

<<<<<<< SEARCH
                average.lerp_(tensor, 0.005)
=======
                average.lerp_(tensor, 0.0025)
>>>>>>> REPLACE