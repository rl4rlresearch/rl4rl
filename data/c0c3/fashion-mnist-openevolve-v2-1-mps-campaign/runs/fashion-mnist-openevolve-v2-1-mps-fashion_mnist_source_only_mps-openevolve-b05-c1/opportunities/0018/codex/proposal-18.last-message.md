MECHANISM: Shortened learning-rate warmup

HYPOTHESIS: Reducing warmup from 5% to 2% will exceed 9,240 correct predictions by providing more near-peak-rate updates while preserving the successful 3.3e-3 peak and late cosine decay.

INTENDED_EDIT: Shorten the cosine schedule’s warmup period from about 52 optimizer steps to about 21, with no change to runtime, architecture, augmentation, or inference.

EVIDENCE: Reducing batch size from 128 to 96 improved correctness from 9,204 to 9,239 through additional optimization updates, while raising peak LR to 3.3e-3 gained another correct prediction; shorter warmup tests additional early optimization progress without the timeout risk of batch size 88 or a still-higher peak rate.

<<<<<<< SEARCH
    warmup_fraction = 0.05
=======
    warmup_fraction = 0.02
>>>>>>> REPLACE