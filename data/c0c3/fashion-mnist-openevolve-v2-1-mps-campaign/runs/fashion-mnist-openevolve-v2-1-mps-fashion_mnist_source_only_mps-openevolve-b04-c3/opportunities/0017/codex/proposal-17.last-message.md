MECHANISM: Nonzero-floor cosine optimization

HYPOTHESIS: Retaining 10% of the peak learning rate at the end of the qualified batch-64 schedule will exceed 9,229 correct predictions by making the late optimizer steps useful without increasing runtime or peak-step instability.

INTENDED_EDIT: Restore the qualified batch size of 64 and change cosine decay from zero-terminal to a 2.5e-4 learning-rate floor.

EVIDENCE: Batch size 64 achieved 9,229 correct versus 9,210 at batch size 128, showing that additional optimization updates help; a low cosine floor increases optimization distance without the added steps that caused smaller batches to time out.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE